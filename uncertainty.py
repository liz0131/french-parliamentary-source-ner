"""
Références :
- ContrastiveAL (Margatina et al., 2021)
- Uncertainty sampling(Settles, 2010)
- Boundary-based AL(Tong & Koller, 2001)
- Sentence-BERT: embedding space semantic structure(Reimer & Gurevych, 2019)
"""
import math
import torch
import numpy as np
from sklearn.cluster import KMeans

class UncertaintySampler:
    """
    Regroupe toutes les stratégies AL :
    1. uncertainty_sampling : entropy ou margn sur les prédictions token-level
    2. model_uncertainty : incertitude du modèle
    3. long_entity_score : longueur de la plus longue entité prédite
    4. semantic_uncertainty : sem_score = 4p(1-p) basé sur les voisins labelisés
    5. kmeans_semantic : kmeans sur emebeddings + score sémantique intra-cluster
    6. semantic_kmeans : pré-filtrage sémantique + kemans + sélection intra-cluster
    7. combined_strategy : stratégie combinant les approches model_uncertainty, kmeans_semantic et long_entity
    """
    def __init__(self, ner_model, k: int=10, batch_size: int=16, alpha: float=0.5, beta: float=0.5): 
        self.ner = ner_model
        self.k = k
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self._labeled_embeddings = None
        self._labeled_has_entity = None
    
    # ------ Model uncertainty ------
    def uncertainty_sampling(self, phrase=None, tokens=None, method="entropy", agg="topk", topk=5):
        """
        méthode : entropy, margin
        aggrégation : topk, max, mean
        """
        if self.ner.model is None or self.tokenizer is None : 
            raise RuntimeError("Model not loaded")
        
        if tokens is None:
            if phrase is None:
                raise ValueError("Provide phrase or tokens")
            tokens = self.simple_tokenize(phrase)
        
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
    
    def model_uncertainty(self, unlabeled_data: list, method: str="entropy", agg: str="max", topk: int=5) -> list:
        scores = []
        for ex in unlabeled_data:
            tokens = ex.get("tokens") or self.ner.simple_tokenize(ex.get("text",""))
            _,_,score,_ = self.ner.uncertainty_sampling(
                tokens = tokens,
                method = method,
                agg = agg,
                topk= topk
            )
            scores.append(score)
        return scores
    
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
            
    # ------- helpers ------
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
                    batch_tokens.append(self.ner.simple_tokenize(ex["text"])) # 

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
    
    def labeled_cache(self, labeled_data: list):
        print(f"Building cache for {len(labeled_data)} labeled examples...")
        embeddings = self.sentence_embedding(labeled_data)
        has_entity = np.array([self._has_entity(ex) for ex in labeled_data], dtype = bool)

        self._labeled_embeddings = self._normalize(embeddings)
        self._labeled_has_entity = has_entity

        n_pos = has_entity.sum()
        print(f"{n_pos} examples avec entité, {(~has_entity).sum()} examples sans entité"
              f"(ratio pos={n_pos/len(labeled_data):2f})")
    
    # ------ Semantic uncertainty ------
    def semantic_uncertainty(self, unlabeled_data:list) -> tuple:
        """
        U_score = 4p(1-p),
        p : proportion de voisins labelisés contenant une entité.
        quand p=0.5 (frontière positive/negatif), U_score est à son maximum.

        """
        if self._labeled_embeddings is None:
            raise RuntimeError("Cache vide.")
        
        unl_emb_norm = self._normalize(self.sentence_embedding(unlabeled_data))
        sim_matrix = unl_emb_norm @ self._labeled_embeddings.T
        k_nn = min(self.k,len(self._labeled_has_entity))
        
        scores = []
        p_raws = []
        for sim_row in sim_matrix:
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            scores.append(float(4*p*(1-p)))
            p_raws.append(float(p))
        return scores, p_raws
    
    def maxmin_distance(self, unlabeled_data: list) -> list:
        if self._labeled_embeddings is None :
            raise RuntimeError("Cache Vide.")
        
        unl_emb_norm = self._normalize(self.sentence_embedding(unlabeled_data))

        # max de similarité : [N_unlabeled, B_labeled]
        sim_matrix = unl_emb_norm @ self._labeled_embeddings.T

        # Pour chaque phrase non labelisée, on calcul similarité max avec le voisin le plus proche 
        max_sim = sim_matrix.max(axis=1)
        # scores = 1 - max_sim : on choisi les phrases qui ont les scores les plus élevés ou non 
        div_scores = (1.0 - max_sim).tolist()

        return div_scores
    
    def kmeans_semantic(self, unlabeled_data: list, n_clusters: int=None, random_state: int=42) -> list:
        if self._labeled_embeddings is None :
            raise RuntimeError("Cache Vide.")
        
        n = len(unlabeled_data)

        if n_clusters is None :
            n_clusters = max(1, n//10)

        # éviter d'avoir plus de clusters que les exemples dans pool
        n_clusters = min(n_clusters, n)

        print(f"Kmeans_semantic : k={n_clusters} clusters sur {n} exmeples...")

        unl_emb_norm = self._normalize(self.sentence_embedding(unlabeled_data))

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10) 
        cluster_labels = kmeans.fit_predict(unl_emb_norm)

        sim_matrix = unl_emb_norm @ self._labeled_embeddings.T
        k_nn = min(self.k, len(self._labeled_has_entity))

        sem_scores = []
        p_raws = []
        for sim_row in sim_matrix : 
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            sem_scores.append(float(4*p*(1-p)))
            p_raws.append(float(p))
        sem_scores = np.array(sem_scores)

        final_scores = np.zeros(n)
        for cluster_id in range(n_clusters):
            cluster_idx = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_idx) == 0:
                continue
            best_local = np.argmax(sem_scores[cluster_idx])
            best_global = cluster_idx[best_local]
            final_scores[best_global] = max(sem_scores[best_global], 1e-9)
        
        n_selected = (final_scores>0).sum()
        print(f"Kmeans_semantic : {n_selected} exemples retenus.")
        return final_scores.tolist(), p_raws
    
    def semantic_kmeans(self, unlabeled_data: list, n_clusters: int = None, beta: float = 5.0, intra_cluster: str="semantic", random_state: int=42) -> list :
        """
        semantic pre-filtrer -> k-means -> sélection intra-cluster(semantic/center/random)
        (Diverse mini-batch AL - Zhdanov, 2019)
        """
        if self._labeled_embeddings is None :
            raise RuntimeError("Cache Vide.")
        
        if intra_cluster not in ("semantic", "center", "random"):
            raise ValueError("intra_cluster non reconnu")
        
        n = len(unlabeled_data)

        if n_clusters is None :
            n_clusters = max(1,n//10)
        n_clusters = min(n_clusters,n)

        n_prefilter = min(int(beta*n_clusters),n)

        print(f"semantic_kmeans pre_filtrer: top {n_prefilter}"
              f"beta={beta}, k={n_clusters}, intra={intra_cluster} sur {n} exemples...")
        
        unl_emb_norm = self._normalize(self.sentence_embedding(unlabeled_data))

        sim_matrix = unl_emb_norm @ self._labeled_embeddings.T
        k_nn = min(self.k, len(self._labeled_has_entity))

        sem_scores = []
        p_raws = []
        for sim_row in sim_matrix :
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            sem_scores.append(float(4*p*(1-p)))
            p_raws.append(float(p))

        sem_scores = np.array(sem_scores)

        top_filter_idx = np.argpartition(sem_scores, -n_prefilter)[-n_prefilter:]
        filtered_emb = unl_emb_norm[top_filter_idx]

        print(f"semantic_kmeans : {n_clusters} clusters sur {len(filtered_emb)} exemples filtrés")

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(filtered_emb)
        cluster_centers = kmeans.cluster_centers_

        rng = np.random.default_rng(random_state)
        final_scores = np.zeros(n)

        for cluster_id in range(n_clusters):
            cluster_local_idx = np.where(cluster_labels==cluster_id)[0]
            if len(cluster_local_idx)==0:
                continue 

            if intra_cluster == "semantic":
                best_local = np.argmax(sem_scores[top_filter_idx[cluster_local_idx]])
                chosen_global = top_filter_idx[cluster_local_idx[best_local]]

            elif intra_cluster == "center":
                center = cluster_centers[cluster_id]
                dists = np.linalg.norm(
                    filtered_emb[cluster_local_idx]-center, axis=1
                )
                best_local = np.argmin(dists)
                chosen_global = top_filter_idx[cluster_local_idx[best_local]]                

            else : # random
                best_local = rng.integers(0, len(cluster_local_idx))
                chosen_global = top_filter_idx[cluster_local_idx[best_local]]

            final_scores[chosen_global] = max(sem_scores[chosen_global], 1e-9)
            # final_scores[chosen_global] = sem_scores[chosen_global]

        n_selected = (final_scores>0).sum()
        print(f"semantic_kmeans: {n_selected} exemples retenus.")

        return final_scores.tolist(), p_raws

    def combined_strategy(self, unlabeled_data: list,
                          n_positive: int = 128,
                          n_negative: int = 72,
                          n_prefilter: int = 1000,
                          n_clusters: int = None,
                          kmeans_ratio: float = 0.6,
                          method: str = "entropy",
                          random_state: int = 42) -> dict :
        """
        AL stratégie combinée :
        
        Positifs (n_positive exemples) :
        1. Pré-filtrage par entropy -> top n_prefilter candidats 
        2. kmeans_semantic sur candidats filtrés -> kmeans_ratio * n_positive exemples
        3. long_entity dans candidats restants filtrés restants aprè kmeans_semantic
            -> (1-kmeans_ratio)*n_positive exempls

        Négatifs (n_negative exemples) :
        4. Sélectionne par p_raw décroissant parmi les phrases pré-filtrés et prédites sans entité
            - p_raw éleve = exemple sémantiquement proche de la frontière positif/negatif
            (réutilse des embedding de l'étape 2 pour positifs sans calcul supplémentaire)

        retour un dict avec les indices dans unlabeled_data.
        """
        if self._labeled_embeddings is None:
            raise RuntimeError("Cache vide.")
        
        n = len(unlabeled_data)
        n_kmeans = int(n_positive*kmeans_ratio)
        n_long = n_positive - n_kmeans 

        # Étape 1 : pré-filtrage entropy
        print(f"Combined AL strategy : pré-filtrage entropy top {n_prefilter} sur {n} exemples ...")
        entropy_scores = []
        pred_labels_all = []
        for ex in unlabeled_data:
            tokens = ex.get("tokens")or self.ner.simple_tokenize(ex.get("text",""))
            _, preds, score, _= self.predict_with_uncertainty(tokens=tokens, method=method, agg="max")
            entropy_scores.append(score)
            pred_labels_all.append(preds)

        sorted_by_entropy = np.argsort(entropy_scores)[::-1]
        top_entropy_idx = sorted_by_entropy[:n_prefilter].tolist()
        top_entropy_data = [unlabeled_data[i] for i in top_entropy_idx]

        # Étape 2 : kmeans_semantic sur les candidats pré-filtrés
        print(f"Combined AL strategy : kmeans_semantic sur {len(top_entropy_data)} candidats => k{n_kmeans} exemples ...")
        n_clusters_km = n_clusters if n_clusters is not None else max(1, n_kmeans)
        
        top_entropy_emb_norm = self._normalize(self.sentence_embedding(top_entropy_data))
        
        kmeans_model = KMeans(n_clusters=n_clusters_km, random_state=random_state, n_init=10)
        cluster_labels_km = kmeans_model.fit_predict(top_entropy_emb_norm)

        sim_top = top_entropy_emb_norm @ self._labeled_embeddings.T
        k_nn = min(self.k, len(self._labeled_has_entity))

        sem_scores_top = []
        for sim_row in sim_top:
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            sem_scores_top.append(float(4*p*(1-p)))
        sem_scores_top = np.array(sem_scores_top)

        km_final_scores = np.zeros(len(top_entropy_data))
        for cluster_id in range(n_clusters_km):
            cluster_idx = np.where(cluster_labels_km == cluster_id)[0]
            if len(cluster_idx)==0:
                continue
            best_local = np.argmax(sem_scores_top[cluster_idx])
            best_global_local = cluster_idx[best_local]
            km_final_scores[best_global_local] = max(sem_scores_top[best_global_local], 1e-9)

        km_selected_local = np.where(km_final_scores > 0)[0].tolist()
        km_selected_global = set(top_entropy_idx[i] for i in km_selected_local)

        # Étape 3 : long_entity dans les candidats restants 
        remain_idx = [i for i in top_entropy_idx if i not in km_selected_global]

        long_scored = []
        for i in remain_idx:
            score = self.long_entity_score(pred_labels_all[i])
            long_scored.append((score, i))
        long_scored.sort(key=lambda x: x[0], reverse=True)
        long_selected = [i for _, i in long_scored[:n_long]]
        # Positive : km_semantic + long_entity 
        positive_idx = list(km_selected_global)+long_selected
        positive_idx_set = set(positive_idx)
    
        # Étape 4 : négatifs par p_raw 
        print(f"Combined AL strategy : sélection de {n_negative} négatofs par p_raw ...")

        neg_candidates = []
        for local_i, global_i in enumerate(top_entropy_idx):
            if global_i in positive_idx_set:
                continue 
            preds = pred_labels_all[global_i]
            if all(p == 'O' for p in preds):
                top_k_idx = np.argpartition(sim_top[local_i], -k_nn)[-k_nn:]
                p_raw = float(self._label_has_entity[top_k_idx].mean())
                neg_candidates.append((p_raw, global_i))

        neg_candidates.sort(key=lambda x:x[0], reverse=True)
        negative_idx = [i for _, i in neg_candidates[:n_negative]]

        print(f"Combined AL Strategy : {len(positive_idx)} positifs + {len(negative_idx)} négatifs = {len(positive_idx)+len(negative_idx)} exemples total"
              f"(kmeans_semantic : {len(km_selected_global)} exemples, long_entity : {len(long_selected)} exemples)")
        
        return {
            "positive_idx": positive_idx,
            "negative_idx": negative_idx,
            "all_idx": positive_idx + negative_idx,
            "km_idx": list(km_selected_global),
            "long_idx": long_selected
        }


    def get_score(self, unlabeled_data: list, strategy: str="semantic", 
                  method: str="entropy", agg: str="max", 
                  n_clusters:int=None, beta: float = 5.0, intra_cluster:str="semantic") -> dict:

        prior = float(self._labeled_has_entity.mean())
        if strategy == "semantic":
            sem_score, p_raws = self.semantic_uncertainty(unlabeled_data)
            return {"Combined score": sem_score, "Semantic score": sem_score, "Model score": [], "P score": p_raws, "Prior": prior}

        if strategy == "entropy":
            model_score = self.model_uncertainty(unlabeled_data, method=method, agg=agg)
            return {"Combined score": model_score, "Semantic score": [], "Model score": model_score}
        
        if strategy == "combined":
            sem_score = self.semantic_uncertainty(unlabeled_data)
            model_score = self.model_uncertainty(unlabeled_data, method=method, agg=agg)
            combined_score = [self.alpha*s+self.beta*e for s, e in zip(sem_score,model_score)]
            return {"Combined score": combined_score, "Semantic score": sem_score, "Model score": model_score}
        
        if strategy == "maxmin" :
            div_score = self.maxmin_distance(unlabeled_data)
            return {"Combined score": div_score, "Diversity score": div_score}
        
        if strategy == "kmeans_semantic":
            km_score, p_raws = self.kmeans_semantic(unlabeled_data, n_clusters=n_clusters)
            return {"Combined score": km_score, "Kmeans-Semantic score": km_score, "P score": p_raws, "Prior": prior}
        
        if strategy == "semantic_kmeans":
            sk_score, p_raws = self.semantic_kmeans(
                unlabeled_data, n_clusters=n_clusters,
                beta=beta, intra_cluster=intra_cluster
            )
            return {"Combined score":sk_score, "Semantic-Kmeans score": sk_score, "P score": p_raws, "Prior": prior}

        raise ValueError("Stratégie inconnue.")
