import math
import torch
import numpy as np
from sklearn.cluster import KMeans

class ALStrategies: 
    def __init__(self, ner_model, k: int=10, batch_size: int=16): 
        self.ner = ner_model
        self.k = k
        self.batch_size = batch_size

        self._labeled_embeddings = None
        self._labeled_has_entity = None

    def run(self, unlabeled_data: list, strategy: str = "kmeans_fp", 
            n_positive: int=64, n_negative: int=36, n_prefilter: int=1000, n_clusters: int=None, 
            kmeans_ratio: float=0.75, method: str="entropy", random_state: int=42) -> dict : 
        if strategy not in ("kmeans_long", "kmeans_fp"):
            raise ValueError(f"Entrez la bonne AL stratégie.")
        if strategy == "kmeans_long":
            step3 ="long_entity"
        else :
            step3 ="fp_score"
        return self.combined_strategy(
            unlabeled_data, n_positive, n_negative, n_prefilter, 
            n_clusters, kmeans_ratio, method, random_state, step3
        )
        
    def labeled_cache(self, labeled_data: list):
        print(f"Building cache for {len(labeled_data)} labeled examples...")
        embeddings = self.sentence_embedding(labeled_data)
        has_entity = np.array([self._has_entity(ex) for ex in labeled_data], dtype = bool)

        self._labeled_embeddings = self._normalize(embeddings)
        self._labeled_has_entity = has_entity

        n_pos = has_entity.sum()
        print(f"{n_pos} examples avec entité, {(~has_entity).sum()} examples sans entité"
              f"(ratio pos={n_pos/len(labeled_data):2f})")
    
    def combined_strategy(self, unlabeled_data, n_positive, n_negative,
                  n_prefilter, n_clusters, kmeans_ratio, method, random_state, step3):
        """
        Stratégie AL combinée — étapes 1, 2, 4 communes, étape 3 paramétrable.

        Étape 1 : entropy + séparation pos/neg + pré-filtrage
        Étape 2 : kmeans_semantic sur positifs  (kmeans_ratio * n_positive)
        Étape 3 : complément positifs selon step3 :
            - "long_entity" : entités longues prédites   (kmeans_long)
            - "fp_score"    : FP potentiels              (kmeans_fp)
        Étape 4 : kmeans_semantic sur négatifs
        """
        if self._labeled_embeddings is None:
            raise RuntimeError("Cache vide.")
        if step3 not in ("long_entity", "fp_score"):
            raise ValueError("Étape3 doit être long_entity ou fp_score")

        n_kmeans = int(n_positive * kmeans_ratio)
        n_extra  = n_positive - n_kmeans

        # --- Étape 1 ---
        entropy_scores, pred_labels_all = self._compute_entropy(unlabeled_data, method)
        pos_candidates, neg_candidates  = self._split_pos_neg(entropy_scores, pred_labels_all)
        top_pos_idx, top_neg_idx, top_pos_data, top_neg_data = self._prefilter(
            unlabeled_data, pos_candidates, neg_candidates,
            n_prefilter, n_positive, n_negative
        )

        k_nn = min(self.k, len(self._labeled_has_entity))
        km_positive = set()
        extra_selected = []
        positive_idx = []

        # --- Étape 2 ---
        if top_pos_data:
            km_pos_list, _ = self.kmeans_semantique(
                top_pos_data, top_pos_idx, n_kmeans, n_clusters, k_nn, random_state
            )
            km_positive = set(km_pos_list)
            remain_pos_idx = [i for i in top_pos_idx if i not in km_positive]

            # --- Étape 3 ---
            if step3 == "long_entity":
                scored = [(self._long_entity_score(pred_labels_all[i]), i)
                          for i in remain_pos_idx]
                scored.sort(key=lambda x: x[0], reverse=True)
                extra_selected = [i for _, i in scored[:n_extra]]

            elif step3 == "fp_score":
                labeled_neg_emb = self._labeled_embeddings[~self._labeled_has_entity]
                if remain_pos_idx and labeled_neg_emb.shape[0] > 0:
                    remain_data     = [unlabeled_data[i] for i in remain_pos_idx]
                    remain_emb_norm = self._normalize(self.sentence_embedding(remain_data))
                    sim_to_neg      = remain_emb_norm @ labeled_neg_emb.T
                    max_sim_to_neg  = sim_to_neg.max(axis=1)
                    scored = sorted(zip(max_sim_to_neg.tolist(), remain_pos_idx), reverse=True)
                    extra_selected = [i for _, i in scored[:n_extra]]
                    print(f"[{step3}] top3 sim_to_neg: {[round(s,3) for s,_ in scored[:3]]}")
                else:
                    extra_selected = remain_pos_idx[:n_extra]
                    print(f"[{step3}] fallback sur remain_pos_idx")

            positive_idx = list(km_positive) + extra_selected

        print(f"[{step3}] Positifs : kmeans={len(km_positive)}, {step3}={len(extra_selected)}")

        # --- Étape 4 ---
        negative_idx, _ = self.kmeans_semantique(
            top_neg_data, top_neg_idx, n_negative, n_clusters, k_nn, random_state
        )

        print(f"[{step3}] Total : {len(positive_idx)} pos + {len(negative_idx)} neg "              f"= {len(positive_idx)+len(negative_idx)} exemples")

        return {
            "positive_idx":    positive_idx,
            "negative_idx":    negative_idx,
            "all_idx":         positive_idx + negative_idx,
            "km_idx":          list(km_positive),
            "extra_idx":       extra_selected,
            "pred_labels_all": pred_labels_all,
            "entropy_scores":  entropy_scores,
            "strategy":        step3
        }


    # Méthodes de sélection principales
    def kmeans_semantique(self, data, idx, n_select, n_clusters, k_nn, random_state):
        if not data :
            return [], np.array([])
    
        n_clusters = n_clusters if n_clusters is not None else max(1, n_select)
        n_clusters = min(n_clusters, len(data))
        
        emb_norm = self._normalize(self.sentence_embedding(data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(emb_norm)

        sim = emb_norm @ self._labeled_embeddings.T
        sem_scores = []
        for sim_row in sim:
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            sem_scores.append(float(4 * p * (1 - p)))
        sem_scores = np.array(sem_scores)

        km_scores = np.zeros(len(data))
        for cluster_id in range(n_clusters):
            cluster_idx = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_idx) == 0:
                continue
            best_local = np.argmax(sem_scores[cluster_idx])
            best_local_global = cluster_idx[best_local]
            km_scores[best_local_global] = max(sem_scores[best_local_global], 1e-9)

        selected_local = np.where(km_scores > 0)[0].tolist()
        selected_global = [idx[i] for i in selected_local[:n_select]]
        return selected_global, emb_norm

    def uncertainty_sampling(self, tokens, method="entropy", agg="topk", topk=5):
        """
        méthode : entropy, margin
        aggrégation : topk, max, mean
        """
        if self.ner.model is None or self.ner.tokenizer is None : 
            raise RuntimeError("Model not loaded")
        
        encoded = self.ner.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            padding=True,
            is_split_into_words=True
        )
        word_ids = encoded.word_ids(batch_index=0) 
        inputs = {k:v.to(self.ner.device) for k,v in encoded.items()}

        with torch.no_grad():
            logits = self.ner.model(**inputs).logits[0] # [seq_len, num_labels]
            probs = torch.softmax(logits, dim=-1)

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
        word_preds = [self.ner.id2label.get(i,"O") for i in pred_ids]
        # Score d'incertitude par token
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
    
    # Helpers 
    def _compute_entropy(self, unlabeled_data, method):
        entropy_scores = []
        pred_labels_all = []
        for ex in unlabeled_data:
            tokens = ex.get("tokens") or self.ner.simple_tokenize(ex.get("text",""))
            _,preds,score,_ = self.uncertainty_sampling(tokens=tokens, method=method,agg="max")
            entropy_scores.append(score)
            pred_labels_all.append(preds)
        return entropy_scores, pred_labels_all
    
    def _split_pos_neg(self, entropy_scores, pred_labels_all):
        "Sépare exemples positifs/negatifs selon préditions du modèle"
        pos_candidates = []
        neg_candidates = []
        for i, (score, preds) in enumerate(zip(entropy_scores, pred_labels_all)):
            if any(pred != "O" for pred in preds):
                pos_candidates.append((score, i))
            else:
                neg_candidates.append((score, i))
        pos_candidates.sort(key=lambda x: x[0], reverse=True)
        neg_candidates.sort(key=lambda x: x[0], reverse=True)
        print(f"Pool : {len(pos_candidates)} positifs +{len(neg_candidates)} négatifs")
        return pos_candidates, neg_candidates
    
    def _prefilter(self, unlabeled_data, pos_candidates, neg_candidates,
                   n_prefilter, n_positive, n_negative):
        n_prefilter_pos = int(n_prefilter*(n_positive/(n_positive+n_negative)))
        n_prefilter_neg = n_prefilter - n_prefilter_pos

        top_pos_idx = [i for _, i in pos_candidates[:n_prefilter_pos]]  # garde tous les positifs si moins que n_prefilter_pos [:max(n_prefilter_pos, len(pos_candidates))]]
        top_neg_idx = [i for _, i in neg_candidates[:n_prefilter_neg]]
        top_pos_data = [unlabeled_data[i] for i in top_pos_idx]
        top_neg_data = [unlabeled_data[i] for i in top_neg_idx]

        print(f"Pool pré-filtré : {len(top_pos_data)} positifs + {len(top_neg_data)} négatifs")
        return top_pos_idx, top_neg_idx, top_pos_data, top_neg_data

    
    def sentence_embedding(self, data:list) -> np.ndarray:
        """
        Embeddings : convertit les exemples en vecteurs de phrases 
        Utilise seulement l'encodeur de CamemBERT(model.roberta) pour une représentation vectorielle,
        retourne un vecteur par phrase par mean pooling masqué
        """
        model = self.ner.model
        tokenizer = self.ner.tokenizer
        device = self.ner.device

        model.eval()
        embeddings = []

        # Traitement par mini-batches 
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i: i + self.batch_size] 

            # Récupère les tokesn de chaque phrase 
            batch_tokens = []
            for ex in batch_data:
                # si déjà tokenisé mot par mot 
                if "tokens" in ex: 
                    batch_tokens.append(ex["tokens"]) 
                # sinon on tokenise la phrase brute via tokenisation naïve par espaces 
                else:
                    batch_tokens.append(self.ner.simple_tokenize(ex["text"]))  

            # encodage par tokenizer CamemBERT
            encoded = tokenizer(
                batch_tokens, 
                is_split_into_words=True,
                truncation=True,  # tronque les phrases trop longues 
                padding=True,     # aligne les phrases d'un batch à la même longueur avec padding
                max_length=self.ner.max_length,
                return_tensors="pt"  
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                # on récupère la représentation contextuelle par l'encodeur seul
                outputs = model.roberta(
                    input_ids = input_ids, 
                    attention_mask = attention_mask,
                    output_hidden_states = False
                )
                # sortie de la dernière couche Transformer 
                # last_hiddden_state contient une représentaion contextuelle par token: [batch, seq_len, hidden]
                hidden = outputs.last_hidden_state      # [B，L，H]

            # mean pooling masqué
            # retourne un seul vecteur par phrase 
            mask = attention_mask.unsqueeze(-1).float() #[B,L]->[B,L,1] 
            # somme pondérée : on met à zéro les vecteurs de padding 
            summed = (hidden*mask).sum(dim=1)           # [B,H]
            counts = mask.sum(dim=1).clamp(min=1e-9)    # division par le nombre de vrais tokens (sans padding)
            sentence_embedding = (summed/counts).cpu().numpy()

            embeddings.append(sentence_embedding)  # vecteur moyen de la phrase [B, H]
        
        return np.vstack(embeddings)  # emplie les batches -> [N_phrases,Hidden_size]
    
    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings,axis=1, keepdims=True).clip(min=1e-9)
        return embeddings/norms
    
    @staticmethod
    def _has_entity(example: dict) -> bool :
        return any(lab in ("B-SRC", "I-SRC", "B", "I") for lab in example.get("labels", []))
    
    @ staticmethod
    def long_entity_score(pred_labels: list) -> float:
        """
        Score basé sur la longueur de la plus longue entité prédite.
        Retourne un score entre [0,1].
        """
        if not pred_labels :
            return 0.0
        max_len = 0
        current_len = 0

        for lab in pred_labels :
            if lab == "B-SRC":
                current_len = 1
            elif lab == "I-SRC":
                current_len += 1
            else : 
                max_len = max(max_len, current_len)
                current_len = 0 
        # si entité couvre jusqu'à la fin de la phrase 
        return float(max(max_len, current_len)) 
            