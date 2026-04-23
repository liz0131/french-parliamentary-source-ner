
import os
import al_loop
from camembert_model import CamembertNER

def run_train():
    TRAIN_FILE = "./dataset/train.json"
    VAL_FILE = "./dataset/valid.json"
    MODEL_DIR = "./IOB_model"

    EPOCHS = 15
    BATCH_SIZE = 8
    LR = 2e-5
    MAX_LENGTH = 128

    ner = CamembertNER(max_length=MAX_LENGTH)

    train_data = ner.read_jsonl(TRAIN_FILE)
    val_data = ner.read_jsonl(VAL_FILE)

    train_data = ner.convert_to_typed_iob(train_data)
    val_data = ner.convert_to_typed_iob(val_data)

    print(f"Loaded {len(train_data)} training examples.")
    print(f"Loaded {len(val_data)} validation examples.")

    best = ner.train(
        train_data=train_data,
        val_data=val_data,
        model_dir=MODEL_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
    )

    print(f"Done. Best F1 micro: {best:.4f}")


def run_predict():
    MODEL_DIR = "./IOB_model"
    MAX_LENGTH = 128

    path = input("Chemin du fichier à prédire (json/jsonl) > ").strip()
    if not os.path.isfile(path):
        print("Fichier introuvable.")
        return

    ner = CamembertNER(max_length=MAX_LENGTH)
    ner.load(MODEL_DIR)
    print("Model loaded.")

    out = ner.predict_jsonl(path)
    print(f"Predictions saved to {out}")


def run_al():
    """
    Lance la boucle AL.
    Définis les chemins et les huperparamètres pour al_loop
    """
    config = {
        # Chemins 
        "MODEL_DIR":      "./IOB_model_1",
        "BEST_MODEL_DIR": "./best_model",
        "LABELED":        "./dataset/train.json",
        "VAL":            "./dataset/valid.json",
        "POOL":           "./dataset/test_pool.json",
        "TO_ANNOTATE":    "./dataset/to_annotate.json",
        "ANNOTATED":      "./dataset/annotated.json",
        # Hyperparamètres retrain
        "MAX_LENGTH":     128,
        "EPOCHS":         15,
        "BATCH_SIZE":     8,
        "LR":             1e-5,
        "PATIENCE":       5,
        # Hyperparamètres AL loops
        "AL_ITERS":       3,
        "K_UNCERTAIN":    35,
        "K_FN_UNCERTAIN": 5,
        "N_PREFILTER":    2000,
        "KMEANS_RATIO":   0.6,
        "AL_STRATEGY":    "kmeans_fp",   # combined|entropy|semantic|maxmin|kmeans_semantic
        # Hyperparamètres sampler 
        "SAMPLER_K":      10,
        "SAMPLER_BATCH":  16,
        "SAMPLER_ALPHA":  0.5,
        "SAMPLER_BETA":   0.5,
    }
    al_loop.main(config)


def main():
    print("\n==============================")
    print("  NER CamemBERT – Main Menu")
    print("==============================")
    print("1 - Train modèle de base")
    print("2 - Prédire avec un modèle")
    print("3 - Lancer Active Learning")
    print("q - Quitter")

    while True:
        choice = input("\nChoix > ").strip().lower()

        if choice == "1":
            run_train()
        elif choice == "2":
            run_predict()
        elif choice == "3":
            run_al()
        elif choice == "q":
            print("END")
            break
        else:
            print("Choix invalide. Tapez 1, 2, 3 ou q.")


if __name__ == "__main__":
    main()
