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

def make_uid_set(items): # encore le filtrage avec uid !!
    return set([x.get("uid") for x in items if x.get("uid") is not None])

def remove_selected_from_pool(pool, selected_uid_set): # encore le filtrage avec uid !! 
    return [x for x in pool if x.get("uid") not in selected_uid_set]

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

        


def main():
    MODEL_DIR = "./IOB_model_1"
    BEST_MODEL_DIR = "./best_model"
    # CAND_MODEL_DIR ="./candidate_model"

    LABELED = "./dataset/train.json"
    VAL = "./dataset/valid.json"
    POOL = "./dataset/test_pool.json"

    TO_ANNOTATE = "./dataset/to_annotate.json"
    ANNOTATED = "./dataset/annotated.json"   

    # hyperparams AL
    AL_ITERS = 3         # loop al 
    K_UNCERTAIN = 35 # combien de phrases incertaines
    K_FN_PROBE = 5    # combien de phrases "sans src" à vérifier (FN)
    AL_STRATEGY = "kmeans_semantic"  # "combined"|"entropy"|"semantic"| "maxmin"|"kmeans_semantic"

    ner = CamembertNER(max_length=128)

    labeled = read_jsonl(LABELED)
    val = read_jsonl(VAL)
    pool = read_jsonl(POOL)

    # labels typed (safe)
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

        # 1) train (ou re-train)
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

            # 2) load best model
        else : 
            ner.load(BEST_MODEL_DIR)

        sampler = UncertaintySampler(ner, k=10, batch_size=16, alpha=0.5, beta=0.5)   

        # 3) score pool
        sampler.labeled_cache(labeled)  # labeled embedding cache

        scores_dict = sampler.get_score(pool, strategy=AL_STRATEGY, method="entropy", agg="max", n_clusters=K_UNCERTAIN)
        combined = scores_dict["Combined score"] 

        scored = []
        no_src_candidates = []

        for i, item in enumerate(pool):
            phrase = item.get("text", "")
            if not phrase:
                continue

            tokens = item.get("tokens") or ner.simple_tokenize(phrase)
            _, pred_labels, _, _ = ner.uncertainty_sampling(tokens=tokens)

            entry = (combined[i], item, tokens, pred_labels)

            if all(lab == "O" for lab in pred_labels):
                no_src_candidates.append(entry)
            else:
                scored.append(entry)

        # 4) select uncertain (top-K)
        scored.sort(key=lambda x: x[0], reverse=True)
        picked_uncertain = scored[:K_UNCERTAIN]

        # 5) select FN probe (random parmi "tout O") => stratégie de selection des négatifs à changer !!!
        random.shuffle(no_src_candidates) 
        picked_fn = no_src_candidates[:K_FN_PROBE]

        # 6) export to_annotate.jsonl
        to_annotate = []

        # a) incertain
        for u, item, tokens, pred_labels in picked_uncertain:
            to_annotate.append({
                "uid": item.get("uid"),
                "id": item.get("id"),
                "text": item.get("text"),
                "tokens": tokens,
                "labels_pred": pred_labels,
                "uncertainty": u,
                "mode": "uncertain"  # renommer !!!
            })

        # b) FN probe => stratégie de selection des négatifs à changer !!!
        for u, item, tokens, pred_labels in picked_fn:
            to_annotate.append({
                "uid": item.get("uid"),
                "id": item.get("id"),
                "text": item.get("text"),
                "tokens": tokens,
                "labels_pred": pred_labels,  # souvent tout O
                "uncertainty": u,
                "mode": "fn_probe"
            })

        write_jsonl(TO_ANNOTATE, to_annotate)
        print(f"Saved {len(to_annotate)} items to annotate -> {TO_ANNOTATE}")

        # 7) annotation humaine
        get_annotations(TO_ANNOTATE, ANNOTATED)


        # 8) import annotated (doit contenir tokens + labels gold)
        new_items = read_jsonl(ANNOTATED)
        # on attend format: {"uid":..,"id":..,"tokens":[...],"labels":[...]}
        new_items = ner.convert_to_typed_iob(new_items)

        # 9) update labeled
        labeled_uid = make_uid_set(labeled)
        added = 0
        for ex in new_items:
            if ex.get("uid") not in labeled_uid:
                labeled.append(ex)
                added += 1
        print(f"Added {added} new labeled examples.")

        # 10) remove selected from pool
        selected_uids = set([x.get("uid") for x in to_annotate if x.get("uid") is not None])
        pool = remove_selected_from_pool(pool, selected_uids)
        print(f"Pool updated. Remaining: {len(pool)}")

        # 11) save updated datasets
        write_jsonl(LABELED, labeled)
        write_jsonl(POOL, pool)
        print("Saved updated labeled.jsonl and pool.jsonl")

        # 12) Retrain après annotation
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

if __name__ == "__main__":
    main()
