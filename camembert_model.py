import os
import re
import json
import math
import torch, random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from transformers import CamembertTokenizerFast, CamembertForTokenClassification

from seqeval.metrics import classification_report as seq_classification_report, f1_score as seq_f1_score
import sklearn.metrics as sk_metrics
from sklearn.metrics import precision_recall_fscore_support

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class TokenDataset(Dataset):
    """
    Dataset minimal PyTorch : juste __len__ et __getitem__.
    On lui donne une liste d'exemples déjà encodés (tensors).
    """
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class CamembertNER:
    """
    - charge des JSONL IOB (tokens + labels)
    - encode avec tokenizer
    - train + evaluate
    - save/load
    """

    def __init__(self, model_name="camembert-base", labels=None, max_length=128):
        if labels is None:
            labels = ["B-SRC", "I-SRC", "O"]

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
        """
        Convertit B/I/O -> B-SRC/I-SRC/O.
        Si déjà en B-SRC/I-SRC/O, ça ne change rien.
        """
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
        self.model.to(self.device)

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
                # check !!
                raise ValueError(
                    f"Tokens/labels mismatch: uid={ex.get('uid')} "
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
                    # check !!
                    if wid >= len(labels):
                        raise ValueError(
                            f"Label index out of range: uid={ex.get('uid')} "
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
        """
        Entraîne et sauvegarde le meilleur modèle (selon F1 micro seqeval).
        """
        if self.model is None or self.tokenizer is None:
            self.init_for_train()

        train_encoded = self.encode_data(train_data)
        val_encoded = self.encode_data(val_data)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(TokenDataset(train_encoded), batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(TokenDataset(val_encoded), batch_size=batch_size, shuffle=False)

        # ajout de weight decay pour limiter overfitting
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)    
        weight_map = {"B-SRC": 2.0, "I-SRC": 1.5, "O": 1.0}
        class_weights = torch.tensor(
            [weight_map[l] for l in self.labels], dtype=torch.float
        ).to(self.device)
        loss_fct = CrossEntropyLoss(weight=class_weights, ignore_index=-100)

        best_f1 = -1.0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += float(loss.item())

            print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

            metrics = self.evaluate(val_loader)

            print("\n[Validation | seqeval (entity-level)]")
            print(metrics["seqeval_report"])
            print(f"F1 (micro): {metrics['seqeval_f1_micro']:.4f}")
            print(f"F1 (macro): {metrics['seqeval_f1_macro']:.4f}")

            print("\n[Validation | token-level (B/I séparés)]")
            print(metrics["token_report_bi"])

            # Ajout de early stopping 
            current_f1 = metrics["seqeval_f1_micro"]
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

        self.load(model_dir)
        return best_f1

    def evaluate(self, val_loader):
        self.model.eval()
        true_labels, pred_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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

        return {
            "seqeval_report": seqeval_report,
            "seqeval_f1_micro": f1_micro,
            "seqeval_f1_macro": f1_macro,
            "token_report_bi": token_report_bi,
        }
    
    def evaluate_data(self, val_data, batch_size=8):
        """
        Évalue sur des données brutes (format JSONL).
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
    
    def predict_jsonl(self, input_path, output_path=None):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        results = []

        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                uid = item.get("uid")
                sent_id = item.get("id")
                phrase = item.get("text", "")

                if not phrase:
                    continue

                words = self.simple_tokenize(phrase)

                encoded = self.tokenizer(
                    words,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    is_split_into_words=True
                )

                word_ids = encoded.word_ids(batch_index=0)
                inputs = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    logits = self.model(**inputs).logits

                preds = torch.argmax(logits, dim=-1)[0].tolist()

                word_preds = []
                prev = None
                for i, wid in enumerate(word_ids):
                    if wid is None:
                        continue
                    if wid != prev:
                        word_preds.append(self.id2label.get(preds[i], "O"))
                    prev = wid

                results.append({
                    "uid": uid,
                    "id": sent_id,
                    "text": phrase,
                    "tokens": words,
                    "labels": word_preds
                })

        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(input_path),
                "predictions_iob.json"
            )

        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(results, out, ensure_ascii=False, indent=2)

        return output_path
    
    def uncertainty_sampling(self, phrase=None, tokens=None, method="entropy", agg="topk", topk=5):
        """
        méthode : entropy, margin
        aggrégation : topk, max, mean
        """
        if self.model is None or self.tokenizer is None : 
            raise RuntimeError("Model not loaded")
        
        if tokens is None:
            if phrase is None:
                raise ValueError("Provide phrase or tokens")
            tokens = self.simple_tokenize(phrase)
        
        encoded = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=True
        )
        word_ids = encoded.word_ids(batch_index=0) 
        inputs = {k:v.to(self.device) for k,v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0] # [seq_len, num_labels]
            probs = torch.softmax(logits, dim=-1)
            # probs = torch.softmax(logits, dim=-1)[0]  # [seq_len, num_labels]
            # preds = torch.argmax(probs, dim=-1).tolist()

        # Prend subword de chaque mot
        word_probs=[]
        prev = None
        for i, wid in enumerate(word_ids):
            if wid is None : 
                continue 
            if wid!=prev : 
                word_probs.append(probs[i].detach().cpu())
            prev = wid
        
        if len(word_probs) == 0 :
            return tokens, [], 0.0, []  
        
        word_probs = torch.stack(word_probs, dim=0)
        pred_ids = torch.argmax(word_probs, dim=-1).tolist()
        word_preds = [self.id2label.get(i,"O") for i in pred_ids]

        if method == "entropy":
            eps = 1e-12
            num_labels = word_probs.shape[-1]
            word_uncert = -(word_probs * torch.log(word_probs + eps)).sum(dim=-1)
            word_uncert = (word_uncert / math.log(num_labels)).tolist()   # Normolized entropy to [0,1]
        elif method == "margin":
            # 1 - (p1 - p2)
            top2 = torch.topk(word_probs, k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            word_uncert = (1.0 - margin).tolist()
        else:
            raise ValueError("method must be 'entropy' or 'margin'")
        
        # sentence aggregation
        if agg == "mean":
            score = float(sum(word_uncert) / max(1, len(word_uncert)))
        elif agg == "max":
            score = float(max(word_uncert))
        elif agg == "topk":
            k = min(topk, len(word_uncert))
            score = float(sum(sorted(word_uncert, reverse=True)[:k]) / k)
        else:
            raise ValueError("agg must be 'mean', 'max', or 'topk'")

        return tokens, word_preds, score, word_uncert

