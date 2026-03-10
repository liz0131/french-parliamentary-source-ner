"""
Références :
- ContrastiveAL (Margatina et al., 2021)
- Uncertainty sampling(Settles, 2010)
- Boundary-based AL(Tong & Koller, 2001)
- Sentence-BERT: embedding space semantic structure(Reimer & Gurevych, 2019)
"""
import torch
import numpy as np
from sklearn.cluster import KMeans

class UncertaintySampler:
    """
    Combine 2 types de calcul d'incertitude :
    - Semanctic boundry : sem_score = 4p(1-p)
        -> on mesure si une phrase est dans la frontière du labelspace selon son embedding 
    - Model uncertainty : entropy ou margin 
        -> on mesure si le modèle est incertain sur ses prédictions au niveau de token 
    Combined_score = alpha*sem_score + beta*model_score
    """
    def __init__(self, ner_model, k: int=10, batch_size: int=16, alpha: float=0.5, beta: float=0.5): 
        self.ner = ner_model
        self.k = k
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self._labeled_embeddings = None
        self._labeled_has_entity = None
    
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
        
    def semantic_uncertainty(self, unlabeled_data:list) -> tuple:
        """
        U_score = 4p(1-p)

        """
        if self._labeled_embeddings is None:
            raise RuntimeError("Cache vide.")
        
        unl_emb_norm = self._normalize(self.sentence_embedding(unlabeled_data))

        sim_matrix = unl_emb_norm @ self._labeled_embeddings.T
        k_nn = min(self.k,len(self._labeled_has_entity))
        
        scores = []
        for sim_row in sim_matrix:
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            scores.append(float(4*p*(1-p)))

        return scores
    
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
        for sim_row in sim_matrix : 
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            sem_scores.append(float(4*p*(1-p)))

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
        return final_scores.tolist()
    
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

        sim_matrix = unl_emb_norm @ self._labelded_embedding.T
        k_nn = min(self.k, len(self._labeded_has_entity))

        sem_scores=[]
        for sim_row in sim_matrix :
            top_k_idx = np.argpartition(sim_row, -k_nn)[-k_nn:]
            p = self._labeled_has_entity[top_k_idx].mean()
            sem_scores.append(float(4*p*(1-p)))

        sem_scores = np.array(sem_scores)

        top_filter_idx = np.argpartition(sem_scores, -n_prefilter)[-n_prefilter:]
        filtered_emb = unl_emb_norm[top_filter_idx]

        print(f"semantic_kmeans : {n_clusters} clusters sur {len(filtered_emb)} exemples filtrés")

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_prefict(filtered_emb)
        cluster_centers = kmeans.clusters_centers_

        rng = np.random.default_rng(random_state)
        final_scores = np.zeros(n)

        for cluster_id in range(n_clusters):
            cluster_local_idx = np.where(cluster_labels==cluster_id)[0]
            if len(cluster_local_idx)==0:
                continue 

            if intra_cluster == "semantic":
                best_local = np.argmax(sem_scores[top_filter_idx[cluster_local_idx]])
                chosen_global = top_filter_idx[cluster_local_idx[best_local]]

            # elif intra_cluster == "center":

            # else : # random
            final_scores[chosen_global] = max(sem_scores[chosen_global], 1e-9)
            # final_scores[chosen_global] = sem_scores[chosen_global]

        n_selected = (final_scores>0).sum()
        print(f"semantic_kmeans: {n_selected} exemples retenus.")

        return final_scores.tolist()


    def get_score(self, unlabeled_data: list, strategy: str="semantic", 
                  method: str="entropy", agg: str="max", 
                  n_clusters:int=None, beta: float = 5.0, intra_cluster:str="semantic") -> dict:

        if strategy == "semantic":
            sem_score = self.semantic_uncertainty(unlabeled_data)
            return {"Combined score": sem_score, "Semantic score": sem_score, "Model score": []}

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
            km_score = self.kmeans_semantic(unlabeled_data, n_clusters=n_clusters)
            return {"Combined score": km_score, "Kmeans-Semantic score": km_score}
        
        if strategy == "semantic_kmeans":
            sk_score = self.semantic_kmeans(
                unlabeled_data, n_clusters=n_clusters,
                beta=beta, intra_cluster=intra_cluster
            )
            return {"Combined score":sk_score, "Semantic-Kmeans score": sk_score}
        # if strategy == "maxmin+semantic":
            div_sem_score = self._semantic_score()

        raise ValueError("Stratégie inconnue.")
