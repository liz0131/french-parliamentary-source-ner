import math
import random
import torch
import numpy as np
from sklearn.cluster import KMeans

class ALStrategies: 
    """
    Stratégies AL pour 4 groupes expériementaux :
    Group 1 : random baseline - sélection aléaoitre 
    Group 2 : entropy_kmeans  - préfiltrage entropie + kmeans avec sélection intra-cluster par entropie 
    Group 3 : kmeans_semantic - préfiltrage entropie + kmeans avec sélection par score de frontière sémantique
    Group 4 : kmeans_fp_fn    - préfiltrage entropie (60% kmeans_semantic + 20% FP-score) + 20% FN-score
    """

    STRATEGIES = ("random", "entropy_kmeans", "kmeans_semantic", "kmeans_fp_fn")
    
    def __init__(self, ner_model, k: int=10, batch_size: int=16): 
        self.ner = ner_model
        self.k = k
        self.batch_size = batch_size

        self._labeled_embeddings = None  # Embeddings normalisés du labeled set
        self._labeled_has_entity = None  

    def run(self, unlabeled_data: list, strategy: str = "kmeans_fp_fn", 
            n_positive: int=120, n_negative: int=30, n_prefilter: int=3000, n_clusters: int=None, 
            kmeans_ratio: float=0.60, fn_ratio: float = 0.20, method: str="entropy", random_state: int=42) -> dict : 
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Choisissez une AL stratégie parmi {self.STRATEGIES}.")
        if strategy == "random":
            return self.random_sampling(unlabeled_data, n_positive, n_negative, random_state) 

        if self._labeled_embeddings is None:
            raise RuntimeError("Cache vide.")

        entropy_scores, pred_labels_all = self._compute_entropy(unlabeled_data, method)
        pos_candidates, neg_candidates = self._split_pos_neg(entropy_scores, pred_labels_all)
        top_pos_idx, top_neg_idx, top_pos_data, top_neg_data = self._prefilter(
            unlabeled_data, pos_candidates, neg_candidates, 
            n_prefilter, n_positive, n_negative
        )
        k_nn = min(self.k, len(self._labeled_has_entity))

        fp_idx = []
        fn_idx = []
        entropy_idx = []

        if strategy == "entropy_kmeans":
            positive_idx, km_idx, entropy_idx = self._select_entropy_kmeans_pos(
                top_pos_data, top_pos_idx, entropy_scores, 
                n_positive, n_clusters, random_state
            )
        
        elif strategy == "kmeans_semantic":
            positive_idx, km_idx, entropy_idx = self._select_kmeans_semantic_pos(
                top_pos_data, top_pos_idx, n_positive, 
                n_clusters, k_nn, random_state
            )

        elif strategy == "kmeans_fp_fn":
            fp_ratio = 1.0 - kmeans_ratio - fn_ratio
            if fp_ratio < 0:
                raise ValueError(
                    f"kmeans_ratio({kmeans_ratio}) + fn_ratio({fn_ratio}) > 1.0"
                )
            n_select = n_positive + n_negative
            n_kmeans = int(n_select*kmeans_ratio)
            n_fp = int(n_select*fp_ratio)
            n_fn = n_select - n_kmeans - n_fp


            positive_idx, km_idx, fp_idx, fn_idx = self._select_kmeans_fp_fn(
                unlabeled_data, top_pos_data, top_pos_idx, neg_candidates,
                n_kmeans, n_fp, n_fn, n_clusters, k_nn, random_state
            )

        if strategy == "kmeans_fp_fn":
            negative_idx = fn_idx
        else:
            negative_idx = self._select_entropy_neg(
                top_neg_data, top_neg_idx, entropy_scores, n_negative
            )
        
        print(f"[{strategy}] Total : {len(positive_idx)} positive + {len(negative_idx)} negative"
              f"={len(positive_idx)+ len(negative_idx)} exemples")
    
        return {
            "positive_idx": positive_idx,
            "negative_idx": negative_idx,
            "all_idx": positive_idx+negative_idx,
            "km_idx": km_idx,
            "fp_idx": fp_idx,
            "fn_idx": fn_idx,
            "entropy_idx": entropy_idx,
            "pred_labels_all": pred_labels_all,
            "entropy_scores": entropy_scores,
            "strategy": strategy,
        }
        
    def labeled_cache(self, labeled_data: list):
        print(f"Building cache for {len(labeled_data)} labeled examples...")
        embeddings = self.sentence_embedding(labeled_data)
        has_entity = np.array([self._has_entity(ex) for ex in labeled_data], dtype = bool)

        self._labeled_embeddings = self._normalize(embeddings)
        self._labeled_has_entity = has_entity

        n_pos = has_entity.sum()
        print(f"{n_pos} examples avec entité, {(~has_entity).sum()} examples sans entité"
              f"(ratio pos={n_pos/len(labeled_data):2f})")
    
    def random_sampling(self, unlabeled_data, n_positive, n_negative, random_state):
        """
        sélection aléaoitre.
        """
        rng = random.Random(random_state)
        all_idx = list(range(len(unlabeled_data)))
        rng.shuffle(all_idx)
        selected = all_idx[:n_positive+n_negative]
        positive_idx = selected[:n_positive]
        negative_idx = selected[n_positive:]

        pred_labels_all = [[] for _ in unlabeled_data]
        entropy_scores = [0.0 for _ in unlabeled_data]
        
        print(f"[random] {len(positive_idx)} positifs + {len(negative_idx)} neg sêlectoionnés aléatoirement")
        return {
            "positive_idx": positive_idx,
            "negative_idx": negative_idx,
            "all_idx": positive_idx + negative_idx, 
            "km_idx": [],
            "fp_idx": [],
            "fn_idx": [],
            "entropy_idx": [],
            "pred_labels_all": pred_labels_all, 
            "entropy_scores": entropy_scores, 
            "strategy": "random",
        }

    def _select_entropy_kmeans_pos(
        self, top_pos_data, top_pos_idx, entropy_scores, 
        n_positive, n_clusters, random_state) : 
        """
        Stratégie entropy_kmeans pour exemples positifs:
        kmeans avec sélection intra-cluster par scorre entropie
        """

        if not top_pos_data:
            return [], [], []

        n_clust = min(
            n_clusters if n_clusters is not None else n_positive,
            len(top_pos_data)
        )
        emb_norm = self._normalize(self.sentence_embedding(top_pos_data))
        kmeans = KMeans(n_clusters=n_clust, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(emb_norm)

        selected = []
        for cid in range(n_clust):
            cluster_local_idx = np.where(cluster_labels == cid)[0]
            if len(cluster_local_idx) == 0 :
                continue
            best_local = max(
                cluster_local_idx, 
                key=lambda local_idx: entropy_scores[top_pos_idx[local_idx]]
            )
            selected.append(top_pos_idx[best_local])

        selected_set = set(selected)
        if len(selected) < n_positive:
            remain = [i for i in top_pos_idx if i not in selected_set]
            remain.sort(key=lambda i: entropy_scores[i], reverse=True)
            selected += remain[: n_positive - len(selected)]

        print(f"[entropy_kmeans] {len(selected)} positives sélectionnés")
        return selected[:n_positive], selected[:n_positive], []

    def _select_kmeans_semantic_pos(
        self, top_pos_data, top_pos_idx,
        n_positive, n_clusters, k_nn, random_state):
        km_list, _ = self.kmeans_semantique(
            top_pos_data, top_pos_idx, n_positive, n_clusters, k_nn, random_state
        )
        print(f"[kmenas_semantic] {len(km_list)} positifs selectés")
        return km_list, km_list, []
    
    def _select_kmeans_fp_fn(
        self, unlabeled_data, top_pos_data, top_pos_idx, neg_candidates, 
        n_kmeans, n_fp, n_fn, n_clusters, k_nn, random_state):

        # Branche 1 : kmeans_semantic
        km_list, _ = self.kmeans_semantique(
            top_pos_data, top_pos_idx, n_kmeans, n_clusters, k_nn, random_state
        )
        km_set = set(km_list)

        # Branche 2 : FP
        remain_pos_idx = [i for i in top_pos_idx if i not in km_set]
        fp_selected    = self._select_fp(unlabeled_data, remain_pos_idx, n_fp)
       
        # Branche 3 : FN
        neg_idx_all = [i for _, i in neg_candidates]
        fn_selected = self._select_fn(unlabeled_data, neg_idx_all, n_fn)

        positive_idx = km_list + fp_selected

        print(f"[kmeans_fp_fn] kmeans={len(km_list)}, fp={len(fp_selected)}, fn={len(fn_selected)}")
        return positive_idx, km_list, fp_selected, fn_selected

    def _select_entropy_neg(self, top_neg_data, top_neg_idx, entropy_scores, n_negative):
        if not top_neg_data:
            return []
        n_clust = min(n_negative, len(top_neg_data))
        emb_norm = self._normalize(self.sentence_embedding(top_neg_data))
        kmeans   = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(emb_norm)

        selected = []
        for cid in range(n_clust):
            cluster_local_idx = np.where(cluster_labels == cid)[0]
            if len(cluster_local_idx) == 0 :
                continue 
            best_local = max(
                cluster_local_idx,
                key=lambda local_idx: entropy_scores[top_neg_idx[local_idx]]
            )
            selected.append(top_neg_idx[best_local])
        # Pour compléter la liste negative si nécessaire
        if len(selected) < n_negative:
            selected_set = set(selected)
            remain = [i for i in top_neg_idx if i not in selected_set]
            remain.sort(key=lambda i: entropy_scores[i], reverse=True)
            selected += remain[: n_negative - len(selected)]
        print(f"{len(selected)} négatifs sélectionnées")
        return selected[:n_negative]


    def _select_fp(self, unlabeled_data, remain_pos_idx, n_fp) :
        labeled_neg_emb = self._labeled_embeddings[~self._labeled_has_entity]
        if not remain_pos_idx or labeled_neg_emb.shape[0] == 0:
            return remain_pos_idx[:n_fp]
        
        remain_data     = [unlabeled_data[i] for i in remain_pos_idx]
        remain_emb_norm = self._normalize(self.sentence_embedding(remain_data))
        sim_to_neg      = remain_emb_norm @ labeled_neg_emb.T          # [N_remain, N_neg_labeled]
        max_sim_to_neg  = sim_to_neg.max(axis=1)       

        scored = sorted(zip(max_sim_to_neg.tolist(), remain_pos_idx), reverse=True)
        return [i for _, i in scored[:n_fp]]

    def _select_fn(self, unlabeled_data, neg_idx_all, n_fn):
        labeled_pos_emb = self._labeled_embeddings[self._labeled_has_entity]
        if not neg_idx_all or labeled_pos_emb.shape[0] == 0:
            return neg_idx_all[:n_fn]
        
        neg_data     = [unlabeled_data[i] for i in neg_idx_all]
        neg_emb_norm = self._normalize(self.sentence_embedding(neg_data))
        sim_to_pos     = neg_emb_norm @ labeled_pos_emb.T        # [N_neg, N_pos_labeled]
        max_sim_to_pos  = sim_to_pos.max(axis=1)       

        scored = sorted(zip(max_sim_to_pos.tolist(),neg_idx_all), reverse=True)
        return [i for _, i in scored[:n_fn]]

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
    
            