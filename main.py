import os
import json
import shutil
import al_loop
from camembert_model import CamembertNER


# ================================================================
# Configurations
# ================================================================
# Les hyperparamètrès à modifier pour différentes expériences 
CONFIG = {
    # Chemins
    "MODEL_DIR":      "./model",
    "CURRENT_MODEL_DIR":"./current_model",
    "BEST_MODEL_DIR": "./best_model",

    "LABELED":        "./dataset/train.json",
    "VAL":            "./dataset/valid.json",
    "POOL":           "./dataset/pool.json",
    "TO_ANNOTATE":    "./dataset/to_annotate.json",
    "ANNOTATED":      "./dataset/annotated.json",
    "AL_STATE":       "./dataset/al_state.json",
    # Hyperparamètres retrain
    "MAX_LENGTH":     128, 
    "EPOCHS":         30,    # 15
    "BATCH_SIZE":     8,
    "LR":             2e-5,  # 1e-5 
    "PATIENCE":       5,     # 3
    # Hyperparamètres AL
    "START_MODE":    "cold",   # "cold" | "warm"
    "AL_STRATEGY":   "kmeans_semantic",   # "random" | "entropy_kmeans" | "kmeans_semantic" | "kmeans_fp_fn"
    "N_SELECT":       150,
    "POS_RATIO":      0.80,   # 80% positifs (120) / 20% négatifs (30)
    "KMEANS_RATIO":   0.60,   # Groupe 4 uniquement
    "FN_RATIO":       0.20,   # Groupe 4 uniquement
    "N_PREFILTER":    2000,
    
    # Hyperparamètres sampler
    "SAMPLER_K":      10,
    "SAMPLER_BATCH":  16,
}


# ================================================================
# Actions du menu
# ================================================================

def run_train():
    ner = CamembertNER(max_length=CONFIG["MAX_LENGTH"])
    train_data = ner.read_jsonl(CONFIG["LABELED"])
    val_data   = ner.read_jsonl(CONFIG["VAL"])
    train_data = ner.convert_to_typed_iob(train_data)
    val_data   = ner.convert_to_typed_iob(val_data)

    print(f"Loaded {len(train_data)} training examples.")
    print(f"Loaded {len(val_data)} validation examples.")

    best = ner.train(
        train_data=train_data,
        val_data=val_data,
        model_dir=CONFIG["BEST_MODEL_DIR"],
        epochs=30,
        batch_size=CONFIG["BATCH_SIZE"],
        lr=2e-5,
        patience=CONFIG["PATIENCE"],
    )

    print(f"Modèle de base - meilleur F1 micro: {best:.4f}")

    # Initialisation du modèle courant utilisé pour la sélection AL
    current_dir = CONFIG["CURRENT_MODEL_DIR"]

    if os.path.exists(current_dir):
        shutil.rmtree(current_dir)

    shutil.copytree(CONFIG["BEST_MODEL_DIR"], current_dir)

    print(f"Modèle courant AL initialisé dans {current_dir}")

    

def run_predict():
    path = input("Chemin du fichier à prédire > ").strip()
    if not os.path.isfile(path):
        print("Fichier introuvable.")
        return

    ner = CamembertNER(max_length=CONFIG["MAX_LENGTH"])
    ner.load(CONFIG["BEST_MODEL_DIR"])
    print("Model loaded.")

    out = ner.predict_jsonl(path, batch_size=64)
    print(f"Predictions saved to {out}")


def run_al_selection():
    """
    Phase A — Sélectionne les exemples et exporte to_annotate.json.
    Lance cette phase, envoie to_annotate.json à l'expert,
    puis lance run_al_retrain() après reception des annotations.
    """
    al_loop.run_selection(CONFIG)


def run_al_retrain():
    """
    Phase B — Importe annotated.json et retrain le modèle.
    Prérequis : annotated.json rempli par l'expert.
    """
    al_loop.run_retrain(CONFIG)


def print_al_status():
    """Affiche l'état courant de la boucle AL."""
    state_path = CONFIG["AL_STATE"]
    if not os.path.isfile(state_path):
        print("Aucun état AL trouvé.")
        return
    with open(state_path) as f:
        state = json.load(f)
    print("\n=== État AL ===")
    print(f"  Itération   : {state['iteration']}")
    print(f"  Phase       : {state['phase']}")
    print(f"  Stratégie   : {state.get('strategy', 'N/A')}")
    print(f"  Best F1     : {state['best_f1']:.4f}")
    print(f"  Sélectionnés: {len(state.get('selected_ids', []))} exemples")


# ================================================================
# Menu principal
# ================================================================

def main():
    print("\n==============================")
    print("  NER CamemBERT – Main Menu")
    print("==============================")
    print("1  - Entraîner modèle de base")
    print("2  - Prédire avec le modèle")
    print("3  - AL : Sélectionner exemples  (exporte to_annotate.json)")
    print("4  - AL : Retrain après annotation (importe annotated.json)")
    print("5  - AL : Afficher état courant")
    print("q  - Quitter")

    while True:
        choice = input("\nChoix > ").strip().lower()

        if choice == "1":
            run_train()
        elif choice == "2":
            run_predict()
        elif choice == "3":
            run_al_selection()
        elif choice == "4":
            run_al_retrain()
        elif choice == "5":
            print_al_status()
        elif choice == "q":
            print("END")
            break
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    main()
