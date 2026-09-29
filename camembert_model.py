import os
import re
import json
import torch, random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from transformers import CamembertTokenizerFast, CamembertForTokenClassification

from seqeval.metrics import classification_report as seq_classification_report, f1_score as seq_f1_score
import sklearn.metrics as sk_metrics
from sklearn.metrics import precision_recall_fscore_support

# from postprocess import postprocess
# à décommenter avec ligne 475 pour appliquer postprocess

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class TokenDataset(Dataset):
    """
    Dataset PyTorch qui encapsule une liste d'exemples déjà encodés (tensors).
    """
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class CamembertNER:
    """
    Modèle NER basé sur CamemBERT-base :
    - charge des exemples annotés au format IOB (tokens + labels)
    - encode les données avec avec tokenizer
    - train + evaluate + predict
    - save + load model
    """

    def __init__(self, model_name="camembert-base", labels=None, max_length=128):
        if labels is None:
            labels = ["O", "B-SRC", "I-SRC"]

        self.model_name = model_name
        self.labels = labels
        self.max_length = max_length

        self.label2id = {lab: i for i, lab in enumerate(labels)}
        self.id2label = {i: lab for lab, i in self.label2id.items()}

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")

        self.tokenizer = None
        self.model = None

    # -------- I/O --------
    def read_jsonl(self, path):
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def convert_to_typed_iob(self, data):
        """Convertit B/I/O -> B-SRC/I-SRC/O."""
        conv = {"B": "B-SRC", "I": "I-SRC", "O": "O"}
        out = []
        for ex in data:
            ex = dict(ex)
            ex["labels"] = [conv.get(l, l) for l in ex.get("labels", [])]
            out.append(ex)
        return out

    # -------- Model --------
    def init_for_train(self):
        self.tokenizer = CamembertTokenizerFast.from_pretrained(self.model_name)
        self.model = CamembertForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        self.model.dropout = torch.nn.Dropout(p=0.2) 
        self.model.to(self.device)

    def freeze_layers(self, num_frozen_layers=4):
        """Gèle les n premières couches pour réduire les pramètres entraînable"""
        # Geler les embeddings 
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

        # Geler les n premièrs couches transformer 
        for i, layer in enumerate(self.model.roberta.encoder.layer):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Compter les paramètres entraînables
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Paramètres entraînables: {trainable:,}/{total:,}{100*trainable/total:.1f}%")

    def load(self, model_dir):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model folder '{model_dir}' not found.")
        self.tokenizer = CamembertTokenizerFast.from_pretrained(model_dir)
        self.model = CamembertForTokenClassification.from_pretrained(model_dir)
        # s'assure du mapping labels
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id
        self.model.to(self.device)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    # -------- Encoding --------
    def encode_data(self, data):
        """
        Transforme data -> liste de dicts de tensors pour DataLoader.
        data attend :
          ex["tokens"] : list[str]
          ex["labels"] : list[str]
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is None. Call init_for_train() or load().")

        encoded_examples = []

        for ex in data:
            tokens = ex["tokens"]
            labels = ex["labels"]

            if len(tokens) != len(labels):
                raise ValueError(
                    f"Tokens/labels mismatch: id={ex.get('id')} "  # correction : uid -> id 
                    f"len(tokens)={len(tokens)} len(labels)={len(labels)}"
                )

            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_offsets_mapping=True
            )

            word_ids = encoded.word_ids()
            aligned_labels = []
            prev_word_id = None

            for wid in word_ids:
                if wid is None:
                    aligned_labels.append(-100)
                elif wid != prev_word_id:
                    if wid >= len(labels):
                        raise ValueError(
                            f"Label index out of range: id={ex.get('id')} "  # correction : uid -> id 
                            f"wid={wid} len(labels)={len(labels)}"
                        )
                    aligned_labels.append(self.label2id[labels[wid]])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid

            encoded_examples.append({
                "input_ids": torch.tensor(encoded["input_ids"]),
                "attention_mask": torch.tensor(encoded["attention_mask"]),
                "labels": torch.tensor(aligned_labels),
            })

        return encoded_examples

    # -------- Train / Eval --------
    def train(self, train_data, val_data, model_dir, epochs=10, batch_size=8, lr=3e-5, patience=5):
        """ Entraîne et sauvegarde le meilleur modèle."""
        if self.model is None or self.tokenizer is None:
            self.init_for_train()

        self.freeze_layers(num_frozen_layers=4)

        train_encoded = self.encode_data(train_data)
        val_encoded = self.encode_data(val_data)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(TokenDataset(train_encoded), batch_size=batch_size, shuffle=True, generator=g)
        # ajout d'évaluation du train
        train_loader_eval = DataLoader(TokenDataset(train_encoded), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(TokenDataset(val_encoded), batch_size=batch_size, shuffle=False)

        # ajout de weight decay pour limiter overfitting
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01) 
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, min_lr=1e-6)   
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        best_f1 = -1.0
        no_improve = 0

        history ={"train_f1":[], "val_f1":[], "gap":[], "token_f1":[], "train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()  # Mettre à jour lr après chaque batch
                

                total_loss += float(loss.item())

            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

            history["train_loss"].append(total_loss)
            train_metrics = self.evaluate(train_loader_eval)
            train_f1 = train_metrics["seqeval_f1_micro"]
            print(f"[TRAIN] F1 micro: {train_f1: .4f}")

            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics["seqeval_f1_micro"]
            history["val_loss"].append(val_metrics["val_loss"])
            print(f"[VALID] F1 micro:{val_f1:.4f}")
            print(f"[VALID] Token F1 (B/I): {val_metrics['token_f1_micro']:.4f}")

            # scheduler.step(val_f1)
            # current_lr = optimizer.param_groups[0]['lr']
            # print(f"[LR] {current_lr:.2e}")

            gap = train_f1 - val_f1
            print(f"[Gap train_valid] {gap:+.4f}{'Overfitting!' if gap > 0.15 else ''}")

            print("\n[Validation | seqeval (entity-level)]")
            print(val_metrics["seqeval_report"])
            print("\n[Validation | token-level (B/I séparés)]")
            print(val_metrics["token_report_bi"])

            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)
            history["gap"].append(gap)
            history["token_f1"].append(val_metrics["token_f1_micro"])

            self.save(model_dir + "_last") 
            # Early stopping selon validation seqeval F1
            current_f1 = val_metrics["seqeval_f1_micro"]
            if current_f1>best_f1:
                best_f1=current_f1
                no_improve = 0
                self.save(model_dir)
                print(f"Best model saved to {model_dir} (best F1 micro={best_f1:.4f})")
            else : 
                no_improve += 1
                print(f"No improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best F1 micro={best_f1:.4f}")
                    break
            

        fig, ax = plt.subplots(2,1,figsize=(8, 8), sharex=True)
        ax[0].plot(history["train_f1"], label="Train F1", color="#F97306")
        ax[0].plot(history["val_f1"], label="Val F1", color="#15B01A")
        ax[0].set_ylabel("F1 micro")
        ax[0].legend()
        ax[0].set_ylim(0,1)
        ax[0].set_title("F1 micro (seqeval)")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(history["train_loss"], label="Train Loss", color="#0343DF", marker="o", markersize=3)
        ax[1].plot(history["val_loss"],   label="Val Loss",   color="#E50000", marker="o", markersize=3)
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].legend()
        ax[1].set_title("Cross-Entropy Loss")
        ax[1].grid(True, alpha=0.3)


        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "training_curve.png"), dpi=120)
        plt.close()    
                
        self.load(model_dir)
        return best_f1

    def evaluate(self, val_loader):
        self.model.eval()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        total_loss = 0.0
        true_labels, pred_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fct(
                    outputs.logits.view(-1, self.model.config.num_labels),
                    labels.view(-1)
                )
                total_loss += float(loss.item())
                predictions = torch.argmax(outputs.logits, dim=-1)

                for pred, true in zip(predictions, labels):
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()

                    sent_true, sent_pred = [], []
                    for p, t in zip(pred, true):
                        if t == -100:
                            continue
                        sent_true.append(self.id2label[int(t)])
                        sent_pred.append(self.id2label[int(p)])

                    true_labels.append(sent_true)
                    pred_labels.append(sent_pred)

        seqeval_report = seq_classification_report(true_labels, pred_labels, digits=4)
        f1_micro = seq_f1_score(true_labels, pred_labels, average="micro")
        f1_macro = seq_f1_score(true_labels, pred_labels, average="macro")

        # token report sur B/I 
        flat_true = [lab for sent in true_labels for lab in sent]
        flat_pred = [lab for sent in pred_labels for lab in sent]
        flat_true_bi = [t for t in flat_true if t != "O"]
        flat_pred_bi = [p for t, p in zip(flat_true, flat_pred) if t != "O"]

        token_report_bi = sk_metrics.classification_report(
            flat_true_bi, flat_pred_bi,
            labels=["B-SRC", "I-SRC"],
            digits=4,
            zero_division=0
        )

        _, _, token_f1_micro, _ = precision_recall_fscore_support(
            flat_true_bi, flat_pred_bi,
            labels=["B-SRC", "I-SRC"],
            average="micro",
            zero_division=0
        )

        return {
            "val_loss": total_loss,
            "seqeval_report": seqeval_report,
            "seqeval_f1_micro": f1_micro,
            "seqeval_f1_macro": f1_macro,
            "token_report_bi": token_report_bi,
            "token_f1_micro": float(token_f1_micro)
        }
    
    def evaluate_data(self, val_data, batch_size=8):
        """
        Évalue sur des données brutes (format JSON).
        Encode les données puis appelle evaluate().
        Retourne le F1 micro.
        """
        val_encoded = self.encode_data(val_data)
        val_loader = DataLoader(TokenDataset(val_encoded), batch_size=batch_size, shuffle=False)
        metrics = self.evaluate(val_loader)
        return metrics["seqeval_f1_micro"]

    # -------- Prediction --------
    def simple_tokenize(self, text):
        text = re.sub(r"([,.;:!?()])", r" \1 ", text)
        text = re.sub(r"(['’])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()
    
    def predict_jsonl(self, input_path, output_path=None, batch_size=64):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        items = []

        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("text", ""):
                    items.append(item)

        results = []
        self.model.eval()

        for batch_start in tqdm(range(0, len(items), batch_size), desc="Predicting"):
            batch_items = items[batch_start: batch_start + batch_size]
            batch_words = [
                it.get("tokens")
                if it.get("tokens")
                else self.simple_tokenize(it.get("text", ""))
                for it in batch_items
            ]

            encoded = self.tokenizer(
                batch_words,
                return_tensors="pt",
                truncation=True,
                padding=True,          # padding dynamique : pad au max du batch courant, éventuellement à max_length si troncature
                max_length=self.max_length,
                is_split_into_words=True
            )

            word_ids_batch = [encoded.word_ids(batch_index=i) for i in range(len(batch_items))]
            inputs = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                
            preds_batch = torch.argmax(logits, dim=-1).tolist()  # [B, seq_len]
                # probs = torch.softmax(logits, dim=-1)[0]
            for batch_idx, item in enumerate(batch_items):
                word_ids = word_ids_batch[batch_idx]
                preds = preds_batch[batch_idx]
                b_src_id = self.label2id["B-SRC"]  
                i_src_id = self.label2id["I-SRC"]  

                word_preds = []
                word_b_probs = []
                word_i_probs = []
                prev = None
                for token_idx, wid in enumerate(word_ids):
                    if wid is None:
                        continue
                    if wid != prev:
                        word_preds.append(self.id2label.get(preds[token_idx], "O"))
                        word_b_probs.append(float(probs[batch_idx, token_idx, b_src_id]))
                        word_i_probs.append(float(probs[batch_idx, token_idx, i_src_id]))
                    prev = wid

                n_words = len(batch_words[batch_idx])
                if len(word_preds) < n_words:
                    n_missing = n_words - len(word_preds)
                    word_preds.extend(["O"] * n_missing)
                    word_b_probs.extend([0.0] * n_missing)
                    word_i_probs.extend([0.0] * n_missing)
                
                # word_preds = postprocess(word_preds, word_i_probs, word_b_probs)

                results.append({
                    "uid": item.get("uid"),
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "tokens": batch_words[batch_idx],
                    "labels": word_preds,
                    # "b_probs": word_b_probs,
                    # "i_probs": word_i_probs,   
                })
            
            if (batch_start // batch_size) % 100 == 0:
                print(f"  {batch_start + len(batch_items)}/{len(items)} phrases traitées...")

        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(input_path),
                "predictions_iob.json"
            )

        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(results, out, ensure_ascii=False, indent=2)

        return output_path
    