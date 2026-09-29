import os
import json
import shutil
from datetime import datetime 
from typing import List
from camembert_model import CamembertNER
from al_strategies import ALStrategies

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------ Utils ------
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

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    
def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def make_id_set(items): 
    return set([x.get("id") for x in items if x.get("id") is not None])

def remove_selected_from_pool(pool, selected_id_set): 
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
    lab[start] = "B-SRC"    
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

# ------ Gestion du meilleur modèle ------
def get_best_model(current_f1, best_f1, model_dir, best_model_dir, iteration) : 
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

# ------ Gestion AL état ------
def load_al_state(state_path):
    if os.path.exists(state_path):
        state = read_json(state_path)
        print(f"État AL chargé : iter={state['iteration']}, phase={state['phase']}, best_f1={state['best_f1']:.4f}")
        return state
    return {
        "iteration": 0,
        "phase": "ready",
        "selected_ids": [],
        "best_f1": 0.0,
        "strategy": "kmeans_fp_fn",
        "timestamp": None,
    }

def save_al_state(state_path, state):
    state["timestamp"]=datetime.now().isoformat()
    write_json(state_path, state)
    print(f"Etat AL sauvegardé : iter={state['iteration']} phase={state['phase']}")

# ------ Export to_annotate.json ------
def build_to_annotate(ner, pool, positive_idx, negative_idx, km_idx_set,  fp_idx_set, fn_idx_set, entropy_idx_set, pred_labels_all, entropy_scores) :
    to_annotate = []
    n = len(pool)
    for idx in list(positive_idx) + list(negative_idx):
        if idx >= n:
            raise ValueError(f"idx={idx} hors limites, pool size={n}. "
                             f"Assure-toi que pool.json n'a pas été modifié "
                             f"entre run_selection et build_to_annotate.")

    for idx in positive_idx:
        item = pool[idx]
        tokens = item.get("tokens") or ner.simple_tokenize(item.get("text",""))
        pred_labels = pred_labels_all[idx]
        uncertainty = entropy_scores[idx]

        if idx in km_idx_set:
            mode = "kmeans"
        elif idx in fp_idx_set:
            mode = "fp"
        elif idx in fn_idx_set:
            mode = "fn"
        elif idx in entropy_idx_set:
            mode = "entropy"
        else:
            mode = "random"
            
        to_annotate.append({
            "uid": item.get("uid"),
            "id": item.get("id"),
            "text": item.get("text"),
            "tokens": tokens,
            "pred_labels": pred_labels,
            "uncertainty": uncertainty,
            "mode": mode
        })

    for idx in negative_idx:
        item = pool[idx]
        tokens = item.get("tokens") or ner.simple_tokenize(item.get("text", ""))
        pred_labels = pred_labels_all[idx]
        uncertainty = entropy_scores[idx]
        mode = "fn" if idx in fn_idx_set else "km_negative"

        to_annotate.append({
            "uid": item.get("uid"),
            "id": item.get("id"),
            "text": item.get("text"),
            "tokens": tokens,
            "pred_labels": pred_labels,
            "uncertainty": uncertainty,
            "mode": mode
        })

    return to_annotate

def run_selection(config:dict):
    cfg            = config
    CURRENT_MODEL_DIR = cfg["CURRENT_MODEL_DIR"]
    BEST_MODEL_DIR = cfg["BEST_MODEL_DIR"]
    LABELED        = cfg["LABELED"]
    VAL            = cfg["VAL"]
    POOL           = cfg["POOL"]
    TO_ANNOTATE    = cfg["TO_ANNOTATE"]
    AL_STATE       = cfg.get("AL_STATE", "./dataset/al_state.json")

    N_SELECT      = cfg.get("N_SELECT", 150)
    POS_RATIO     = cfg.get("POS_RATIO", 0.80)
    K_POSITIVE    = int(N_SELECT * POS_RATIO)
    K_NEGATIVE    = N_SELECT - K_POSITIVE
    N_PREFILTER   = cfg.get("N_PREFILTER", 1000)
    KMEANS_RATIO  = cfg.get("KMEANS_RATIO", 0.75)
    STRATEGY      = cfg.get("AL_STRATEGY", "kmeans_semantic")
    SAMPLER_K     = cfg.get("SAMPLER_K", 10)
    SAMPLER_BATCH = cfg.get("SAMPLER_BATCH", 16)

    print(f"N_SELECT={N_SELECT} | stratégie={STRATEGY} | POS_RATIO={POS_RATIO} -> K_POSITIVE={K_POSITIVE}, K_NEGATIVE={K_NEGATIVE}")

    state = load_al_state(AL_STATE)

    if state["phase"]=="to_annotate":
        print(f"En attende d'annotation. Lance run_retrain() après avoir rempli {cfg['ANNOTATED']}")
        return
    
    iteration = state["iteration"]
    print(f"\n==== PHASE A : SELECTION | itération {iteration+1} ====")

    ner = CamembertNER(max_length=cfg["MAX_LENGTH"])
    labeled = read_jsonl(LABELED)
    val = read_jsonl(VAL)
    pool = read_jsonl(POOL)

    labeled = ner.convert_to_typed_iob(labeled)
    val = ner.convert_to_typed_iob(val)

    print(f"Labeled size: {len(labeled)} | Pool size: {len(pool)}")

    if (os.path.isdir(CURRENT_MODEL_DIR) and os.path.isfile(os.path.join(CURRENT_MODEL_DIR, "config.json"))):
        selection_model_dir = CURRENT_MODEL_DIR
        print(f"Modèle chargé depuis {selection_model_dir}")
    elif (os.path.isdir(BEST_MODEL_DIR) and os.path.isfile(os.path.join(BEST_MODEL_DIR, "config.json"))):
        selection_model_dir = BEST_MODEL_DIR
        print(f"Modèle chargé depuis {selection_model_dir}")
    else : 
        print(f"Aucun modèle disponible")
        return

    ner.load(selection_model_dir)
    current_f1 = ner.evaluate_data(val, batch_size=cfg.get("BATCH_SIZE", 8))
    print(f"F1 modèle actuel : {current_f1:.4f}")

    # Sélection AL
    sampler = ALStrategies(ner, k=SAMPLER_K, batch_size=SAMPLER_BATCH)
    sampler.labeled_cache(labeled)

    print(f"Stratégie AL : {STRATEGY}")
    result = sampler.run(
        unlabeled_data=pool,
        strategy=STRATEGY,
        n_positive=K_POSITIVE,
        n_negative=K_NEGATIVE,
        n_prefilter=N_PREFILTER,
        kmeans_ratio=KMEANS_RATIO,
        fn_ratio=cfg.get("FN_RATIO", 0.0),
        method="entropy"
    )        

    positive_idx = result["positive_idx"]
    negative_idx = result["negative_idx"]
    km_idx_set = set(result["km_idx"])
    fp_idx_set = set(result["fp_idx"])
    fn_idx_set = set(result["fn_idx"])
    entropy_idx_set = set(result["entropy_idx"])
    pred_labels_all = result["pred_labels_all"]
    entropy_scores = result["entropy_scores"]

    print(f"Pool size au moment de la sélection : {len(pool)}")
    print(f"Max idx positif : {max(positive_idx) if positive_idx else 'N/A'}")
    print(f"Max idx négatif : {max(negative_idx) if negative_idx else 'N/A'}")

    # Export to_annotate.json
    to_annotate = build_to_annotate(
        ner, pool, positive_idx, negative_idx, 
        km_idx_set, fp_idx_set, fn_idx_set, 
        entropy_idx_set, pred_labels_all, entropy_scores,
    )

    write_jsonl(TO_ANNOTATE, to_annotate)
    selected_ids = [x["id"] for x in to_annotate if x.get("id")]

    print(f"\n{len(to_annotate)} exemples exportes vers {TO_ANNOTATE}")
    print(f"positifs={len(positive_idx)} | negatifs={len(negative_idx)}")
    print(f"En attente d'annotation externe...")
    print(f"Place les annotations dans {cfg['ANNOTATED']}")
    print(f"Lance ensuite run_retrain()")

    # Sauvegarder etat AL
    state.update({
        "iteration": iteration,
        "phase": "to_annotate",
        "selected_ids": selected_ids,
        "strategy": STRATEGY,
    })
    save_al_state(AL_STATE, state)

def run_retrain(config: dict):
    cfg            = config
    MODEL_DIR      = cfg["MODEL_DIR"]
    CURRENT_MODEL_DIR = cfg["CURRENT_MODEL_DIR"]
    BEST_MODEL_DIR = cfg["BEST_MODEL_DIR"]
    LABELED        = cfg["LABELED"]
    VAL            = cfg["VAL"]
    POOL           = cfg["POOL"]
    ANNOTATED      = cfg["ANNOTATED"]
    AL_STATE       = cfg.get("AL_STATE", "./dataset/al_state.json")
    START_MODE     = cfg.get("START_MODE", "cold")

    EPOCHS     = cfg.get("EPOCHS", 15)
    BATCH_SIZE = cfg.get("BATCH_SIZE", 8)
    LR         = cfg.get("LR", 1e-5)
    PATIENCE   = cfg.get("PATIENCE", 3)

    # Charger etat AL
    state = load_al_state(AL_STATE)

    if state["phase"] != "to_annotate":
        print("Aucune annotation en attente. Lance d'abord run_selection().")
        return

    iteration = state["iteration"]
    selected_ids = set(state["selected_ids"])
    best_f1 = state["best_f1"]

    print(f"\n===== PHASE B - RETRAIN (iteration {iteration + 1}) =====")

    # Verifier annotated.json 
    if not os.path.isfile(ANNOTATED):
        print(f"Fichier {ANNOTATED} introuvable.")
        return
    
    # Charger donnees
    ner = CamembertNER(max_length=cfg["MAX_LENGTH"])
    labeled = read_jsonl(LABELED)
    val = read_jsonl(VAL)
    pool = read_jsonl(POOL)

    labeled = ner.convert_to_typed_iob(labeled)
    val = ner.convert_to_typed_iob(val)

    # Import annotations gold
    new_items = read_jsonl(ANNOTATED)
    new_items = ner.convert_to_typed_iob(new_items)

    # Verification
    annotated_ids = make_id_set(new_items)
    missing = selected_ids - annotated_ids
    if missing:
        print(f"{len(missing)} exemples selectionnés non annotés.")
        print(f"IDs manquants : {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # Mise a jour labeled
    labeled_id_set = make_id_set(labeled)
    added = 0
    for ex in new_items:
        if ex.get("id") not in labeled_id_set:
            labeled.append(ex)
            added += 1
    print(f"Ajout de {added} nouveaux exemples. Labeled size : {len(labeled)}")

    # Mise a jour pool
    pool = remove_selected_from_pool(pool, selected_ids)
    print(f"Pool mis a jour. Restant : {len(pool)}")

    # Sauvegarde datasets
    write_jsonl(LABELED, labeled)
    write_jsonl(POOL, pool)
    print("Labeled + pool sauvegardés.")

    # Cold start or Warm start retrain 
    if START_MODE == "warm":
        print(f"\nRetrain (warm start) depuis {BEST_MODEL_DIR}...")
        ner.load(BEST_MODEL_DIR)
    elif START_MODE == "cold":
        print(f"\nRetrain (cold start) — réinitialisation depuis {ner.model_name}...")
        ner.init_for_train()   # train() le ferait aussi si ner.model est None
    else:
        raise ValueError(f"START_MODE inconnu: {START_MODE!r} (attendu 'cold' ou 'warm')")

    retrain_f1 = ner.train(
        train_data=labeled,
        val_data=val,
        model_dir=MODEL_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE
    )
    if os.path.exists(CURRENT_MODEL_DIR):
        shutil.rmtree(CURRENT_MODEL_DIR)

    shutil.copytree(MODEL_DIR, CURRENT_MODEL_DIR)

    print(f"Modèle AL actuel mis à jour depuis {MODEL_DIR} vers {CURRENT_MODEL_DIR}")
    print(f"F1 apres retrain : {retrain_f1:.4f}")

    best_f1 = get_best_model(
        ner, retrain_f1, best_f1, MODEL_DIR, BEST_MODEL_DIR, iteration + 1
    )

    # Mettre à jour état AL
    state.update({
        "iteration": iteration + 1,
        "phase": "ready",
        "selected_ids": [],
        "best_f1": best_f1,
    })
    save_al_state(AL_STATE, state)

    print(f"\nIteration {iteration + 1} terminée.")
    print(f"Meilleur F1 : {best_f1:.4f}")
    print(f"Meilleur modèle : {BEST_MODEL_DIR}")
