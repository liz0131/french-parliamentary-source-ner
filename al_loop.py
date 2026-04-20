import os
import json
import random
import shutil
from typing import List
from camembert_model import CamembertNER
from uncertainty import UncertaintySampler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def make_id_set(items): # correction : référence uid -> id
    return set([x.get("id") for x in items if x.get("id") is not None])

def remove_selected_from_pool(pool, selected_id_set): # correction : référence uid -> id
    return [x for x in pool if x.get("id") not in selected_id_set]

def visual_token_label(tokens: List[str], labels:List[str]) -> str:
    """ Visualisation des annotations pour vérifier tokens/labels """
    lines =[]
    for i,(t,lab) in enumerate(zip(tokens, labels)):
        lines.append(f"{i:>3} {t:<20} {lab}")
    return "\n".join(lines)

def add_span(labels:List[str], start:int, end: int) -> List[str]: # ent: str
    """ Application d'un span [start, end] selon positions du token pour modifier annotation en IOB """
    if start < 0 or end >= len(labels) or start > end:
        raise ValueError("Span invalide.")
    lab = labels[:]
    lab[start] = "B-SRC"    # lab[start] = f"B-{ent}"
    for i in range(start+1, end+1):
        lab[i] = "I-SRC"
    return lab

def clear_span(labels: List[str], start: int, end: int) -> List[str]:
    """ Met un span [start, end] à O pour supprimer les labels """
    if start < 0 or end  >= len(labels) or start > end:
        raise ValueError("Span invalide.")
    lab = labels[:]
    for i in range(start, end+1):
        lab[i] = "O"
    return lab 

def get_best_model(ner, current_f1, best_f1, model_dir, best_model_dir, iteration) : 
    print(f"F1 actuel (iter {iteration}): {current_f1:.4f}")
    print(f"Meilleur F1 précédent: {best_f1:.4f}")

    if current_f1 > best_f1 :
        print(f"Nouveau meilleur modèle sauvegardé dans {best_model_dir}")
        
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        shutil.copytree(model_dir, best_model_dir)

        metadata = {
            "best_f1": current_f1,
            "iteration": iteration
        }

        with open(os.path.join(best_model_dir, "metadata.json"), "w") as f: 
            json.dump(metadata, f, indent=2) 
        return current_f1
    else : 
        print("Le meilleur modèle reste inchangé.")
        return best_f1 
    
def load_best_f1(best_model_dir):
    metadata_path = os.path.join(best_model_dir,"metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            return metadata.get("best_f1", 0.0)
    return 0.0

def get_menu():
    print("\nCommandes :")
    print(" Entrée : valider la phrase avec ses annotations ")
    print(" 0     : supprimer span pour une entité annotée, ex: 3 6(start end)")
    print(" 1      : ajouter span pour modifier une entité, ex : 3 6")
    print(" r      : revenir à la phrase précédente")
    print(" d      : revoir la page de définitions")
    print(" s      : sauvegarder tous les annotations\n")
    print(" Labels attendus : O, B-SRC, I-SRC")
    print(" Indice : 0 based\n")

def get_annotations(to_annotate_path: str, annotated_path:str) -> None : 
    items = read_jsonl(to_annotate_path)
    if not items : 
        print(f"Aucun élément à annoter dans {to_annotate_path}")
        return
    
    labeled = read_jsonl(annotated_path)
    done = {x.get("id"): x for x in labeled if x.get("id") is not None}  

    output = labeled[:]

    print(f"\nLoaded {len(items)} phrases. {len(done)} sont déjà annotées.\n")
    get_menu()

    i = 0 
    while i < len(items):
        item = items[i]
        item_id = item.get("id")
        uid = item.get("uid")
        tokens = item.get("tokens") or []
        pred = item.get("labels_pred") or ["O"]*len(tokens)
        text = item.get("text", "")

        current = (done.get(item_id, {}).get("labels")or pred)[:]   # copie de la liste
        print (f"\n[{i+1}/{len(items)}]  uid={uid}  id={item_id}  uncertainty={item.get('uncertainty')}  mode={item.get('mode')}")  # Doublon uid !!
        if text:
            print(text)
        print("\nTokens, Labels")
        print(visual_token_label(tokens, current))

        cmd = input("\n> ").strip()

        # Entrée : valider
        if cmd=="":
            record = {"uid":uid, "id": item_id, "tokens": tokens, "labels":current}
            output = [r for r in output if r.get("id")!=item_id] 
            output.append(record)
            done[item_id] = record 
            i+=1
            continue
        
        # r : revenir
        if cmd =="r":
            i = max(0, i-1)
            continue 
        
        # d : help menu
        if cmd.lower() == "d":
            get_menu()
            continue 

        # s : sauvegarde 
        if cmd.lower()=="s":
            saved = list(done.values())
            saved.sort(key=lambda x: str(x.get("uid")))
            write_jsonl(annotated_path, saved)
            print(f"\n {len(saved)} phrases annotées sont sauvegardées à {annotated_path}")
            return 
        
        # 1 : modifier src 
        if cmd == "1":
            line = input("Modifiez le span entre (start end)>").strip()
            try: 
                s,e = line.split()
                start, end = int(s), int(e)
                current = add_span(current, start, end)
                done[item_id] = {"uid": uid, "id": item_id, "tokens": tokens, "labels": current}
                print("Span ajouté. Validez avec Entrée.")
            except Exception as ex : 
                print(f"Erreur:{ex}")
            continue 
        
        # 0 : supprimer src 
        if cmd =="0":
            line = input("Delete l'étiquette entre (start end) > ").strip()
            try:
                s, e = line.split()
                start, end = int(s), int(e)
                current = clear_span(current, start, end)
                done[item_id]={"uid":uid, "id":item_id, "tokens": tokens, "labels":current}
                print("L'étiquette supprimée. Validez avec Entrée.")
            except Exception as ex :
                print(f"Erreur: {ex}")
            continue 

        print("Commande inconnue. Tapez 'd' pour avoir les définitions en détails.")

    # Sauvegarde automatique 
    saved = list(done.values())
    saved.sort(key=lambda x: str(x.get("uid")))
    write_jsonl(annotated_path, saved)
    print(f"\n Merci pour l'annotation. Résultats enregistrés à {annotated_path}")

        


def main(config: dict):
    cfg = config 
    MODEL_DIR      = cfg["MODEL_DIR"]
    BEST_MODEL_DIR = cfg["BEST_MODEL_DIR"]
    LABELED        = cfg["LABELED"]
    VAL            = cfg["VAL"]
    POOL           = cfg["POOL"]
    TO_ANNOTATE    = cfg["TO_ANNOTATE"]
    ANNOTATED      = cfg["ANNOTATED"]
    AL_ITERS       = cfg["AL_ITERS"]
    K_POSITIVE     = cfg["K_POSITIVE"]
    K_NEGATIVE     = cfg["K_NEGATIVE"]
    N_PREFILTER    = cfg.get("N_PREFILTER", 2000)
    KMEANS_RATIO   = cfg.get("KMEANS_RATIO", 0.6)

    ner = CamembertNER(max_length=cfg["MAX_LENGTH"])

    labeled = read_jsonl(LABELED)
    val = read_jsonl(VAL)
    pool = read_jsonl(POOL)

    # labels typed
    labeled = ner.convert_to_typed_iob(labeled)
    val = ner.convert_to_typed_iob(val)

    # Charger le meilleur F1 connu
    best_f1 = load_best_f1(BEST_MODEL_DIR)
    best_f1_micro = best_f1
    print(f"Meilleur F1 micro : {best_f1: 4f}")

    # Vérifier si un modèle existe pour warm start
    saved_model = (
        os.path.isdir(MODEL_DIR)
        and os.path.isfile(os.path.join(MODEL_DIR, "config.json"))
    )

    for it in range(AL_ITERS):
        print(f"\n===== AL ITERATION {it+1}/{AL_ITERS} =====")
        print(f"Labeled size: {len(labeled)} | Pool size: {len(pool)}")

        # 1) warm start / entraînement
        if it == 0 :
            if saved_model:
                print(f"Warm start: loading existing model from {MODEL_DIR}")
                ner.load(MODEL_DIR)
                # évaluer le mode1le de warm start
                current_f1 = ner.evaluate_data(val, batch_size=8)
                print(f"F1-score du modèle warm start: {current_f1:.4f}")
                best_f1 = get_best_model(ner, current_f1, best_f1, MODEL_DIR, BEST_MODEL_DIR, it+1)

            else:
                print(f"Entraînement du modèle (itération {it+1})...")
                current_f1 = ner.train(
                    train_data=labeled,
                    val_data=val,
                    model_dir=MODEL_DIR,
                    epochs=15,
                    batch_size=8,
                    lr=1e-5,
                    patience =5
                )
                print(f"Le meilleure F1 micro: {current_f1:.4f}")
                # Comparer et sauvegarder si meilleur 
                best_f1 = get_best_model(ner, current_f1, best_f1, MODEL_DIR, BEST_MODEL_DIR, it+1)
        
        else : 
            ner.load(BEST_MODEL_DIR)

        # 2) Initialisation du sampler
        sampler = UncertaintySampler(ner, k=10, batch_size=16, alpha=0.5, beta=0.5)   
        sampler.labeled_cache(labeled)  # labeled embedding cache

        # 3) Stratégie combiné et score
            # scores_dict = sampler.get_score(pool, strategy=AL_STRATEGY, method="entropy", agg="max", n_clusters=K_UNCERTAIN)
            # combined = scores_dict["Combined score"]
        result = sampler.combined_strategy(
            unlabeled_data=pool,
            n_positive=K_POSITIVE,
            n_negative=K_NEGATIVE,
            n_prefilter=N_PREFILTER,
            kmeans_ratio=KMEANS_RATIO, 
            method="entropy"
        )

        positive_idx = result["positive_idx"]
        negative_idx = result["negative_idx"]
        km_idx_set = set(result["km_idx"])
        long_idx_set = set(result["long_idx"])

        # 4) Export to_annotate.json
        to_annotate = []

        for idx in positive_idx:
            item = pool[idx]
            tokens = item.get("tokens") or ner.simple_tokenize(item.get("text",""))
            _, pred_labels, uncertainty, _ = sampler.predict_with_uncertainty(tokens=tokens, method="entropy", agg="max")
            
            if idx in km_idx_set:
                mode = "kmeans"
            elif idx in long_idx_set:
                mode = "long_entity"
            else:
                mode = "uncertain"
            
            to_annotate.append({
                "uid": item.get("uid"),
                "id": item.get("id"),
                "text": item.get("text"),
                "tokens": tokens,
                "pred_labels": pred_labels,
                "uncertainty": uncertainty,
                "long_score": UncertaintySampler.long_entity_score(pred_labels),
                "mode": mode
            })

        for idx in negative_idx:
            item = pool[idx]
            tokens = item.get("tokens") or ner.simple_tokenize(item.get("text",""))
            _, pred_labels, uncertainty, _ = sampler.predict_with_uncertainty(tokens=tokens, method="entropy", agg="max")
            to_annotate.append({
                "uid": item.get("uid"),
                "id": item.get("id"),
                "text": item.get("text"),
                "tokens": tokens,
                "pred_labels": pred_labels,
                "uncertainty": uncertainty,
                "long_score": 0.0,
                "mode": "p_raw_negative"
            })

        write_jsonl(TO_ANNOTATE, to_annotate)
        print(f"Saved {len(to_annotate)} items to annotate -> {TO_ANNOTATE}")
        print(f"positive={len(km_idx_set)+len(long_idx_set)}"
              f"negative={len(negative_idx)}")
        # 5) Annotation humaine
        get_annotations(TO_ANNOTATE, ANNOTATED)

        # 6) Import des annotations gold
        new_items = read_jsonl(ANNOTATED)
        new_items = ner.convert_to_typed_iob(new_items)

        # 7) Mise à jour du labeled set
        labeled_id = make_id_set(labeled)
        added = 0
        for ex in new_items:
            if ex.get("id") not in labeled_id:
                labeled.append(ex)
                added += 1
        print(f"Added {added} new labeled examples.")

        # 8) Mise à jour du pool
        selected_ids = set([x.get("id") for x in to_annotate if x.get("id") is not None])
        pool = remove_selected_from_pool(pool, selected_ids)
        print(f"Pool updated. Remaining: {len(pool)}")

        # 9) Sauvegarde du datasets mis-à-jour
        write_jsonl(LABELED, labeled)
        write_jsonl(POOL, pool)
        print("Saved updated labeled.jsonl and pool.jsonl")

        # 10) Retrain après annotation
        print(f"Entraînement du modèle (itération {it+1})...")
        retrain_f1 = ner.train(
            train_data=labeled,
            val_data=val,
            model_dir=MODEL_DIR,
            epochs=15,
            batch_size=8,
            lr=1e-5,
            patience = 5
        )
        print(f"Le meilleure F1 micro: {retrain_f1:.4f}")
        best_f1_micro = get_best_model(ner, retrain_f1, best_f1, MODEL_DIR, BEST_MODEL_DIR, it+1)

    print(f"Meilleur F1 global atteint: {best_f1_micro:.4f}")
    print(f"Meilleur modèle sauvegardé dans: {BEST_MODEL_DIR}")
