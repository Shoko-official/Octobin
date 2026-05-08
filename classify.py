"""
classify.py -- Detection en temps reel + apprentissage correctif
----------------------------------------------------------------
Ouvre la webcam, classe le dechet filme, et integre les corrections
de l'utilisateur dans le modele (fine-tuning en ligne).

Usage :
  python classify.py

Controles :
  [ESPACE]  -> Capturer et classifier l'image actuelle
  [1-6]     -> Corriger la classe predite (voir la legende affichee)
  [s]       -> Re-entrainer le modele sur les corrections accumulees
  [q]       -> Quitter
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIG --------------------------------------------------------------------
MODEL_PATH      = "waste_model.keras"
LABELS_PATH     = "labels.txt"
CORRECTIONS_DIR = "corrections"   # Images mal classees sauvegardees ici
IMG_SIZE        = (224, 224)
CONFIDENCE_MIN  = 0.70          # Seuil de confiance minimal (alors 70 c haut oui mais softmax peut etre trop confiant)

# Couleurs par categorie (BGR)
COLORS = {
    "papier":    (100, 200, 255),
    "plastique": (0, 165, 255),
    "verre":     (255, 200, 0),
    "organique": (50, 180, 50),
}
DEFAULT_COLOR = (200, 200, 200)


# --- CHARGEMENT ----------------------------------------------------------------
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERREUR] Modele introuvable : {MODEL_PATH}")
        print("  Lance d'abord : python train.py")
        exit(1)
    if not os.path.exists(LABELS_PATH):
        print(f"[ERREUR] Labels introuvables : {LABELS_PATH}")
        exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        labels = [l.strip() for l in f if l.strip()]
            
    print(f"[SUCCES] Modele charge -- {len(labels)} classes : {labels}")
    return model, labels


# --- PRE-TRAITEMENT ------------------------------------------------------------
def preprocess(frame):
    """Préparation conforme à MobileNetV2"""
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return np.expand_dims(preprocess_input(img), axis=0)


# --- PREDICTION ----------------------------------------------------------------
def predict(model, frame: np.ndarray, labels: list) -> tuple[str, float, np.ndarray]:
    """Retourne (classe, confiance, vecteur de probabilites)."""
    tensor = preprocess(frame)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), probs


# --- SAUVEGARDE D'UNE CORRECTION -----------------------------------------------
def save_correction(frame: np.ndarray, true_label: str) -> str:
    """Sauvegarde l'image dans corrections/<true_label>/."""
    folder = os.path.join(CORRECTIONS_DIR, true_label)
    os.makedirs(folder, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(folder, f"{ts}.jpg")
    cv2.imwrite(path, frame)
    return path



# --- OVERLAY UI ----------------------------------------------------------------
def draw_ui(frame, labels, result=None, state="live"):
    """
    Dessine l'interface sur la frame.
    state : 'live' | 'captured'
    result : (predicted_label, confidence, probs) ou None
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Barre du bas semi-transparente
    cv2.rectangle(overlay, (0, h - 140), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if state == "live":
        cv2.putText(frame, "LIVE -- Appuie sur [ESPACE] pour classifier",
                    (10, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    elif state == "captured" and result:
        pred, conf, probs = result
        color = COLORS.get(pred, DEFAULT_COLOR)

        # Classe predite
        cv2.putText(frame, f"Prediction : {pred.upper()}  ({conf*100:.1f}%)",
                    (10, h - 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Barre de confiance
        bar_w = int((w - 20) * conf)
        cv2.rectangle(frame, (10, h - 88), (10 + bar_w, h - 72), color, -1)
        cv2.rectangle(frame, (10, h - 88), (w - 10, h - 72), (180, 180, 180), 1)

        # Legende correction
        cv2.putText(frame, "Correct ? [O] oui  |  Mauvais ? tape le numero de la vraie classe :",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        for i, lbl in enumerate(labels):
            x = 10 + i * (w // len(labels))
            cv2.putText(frame, f"[{i+1}] {lbl}", (x, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        COLORS.get(lbl, DEFAULT_COLOR), 1)

    # Instructions permanentes
    cv2.putText(frame, "[q] Quitter",
                (w - 110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Reticule central
    cx, cy = w // 2, h // 2
    size = 60
    cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size),
                  (0, 255, 120), 2)

    return frame


# --- BOUCLE PRINCIPALE ---------------------------------------------------------
def run():
    model, labels = load_model_and_labels()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state          = "live"    # 'live' ou 'captured'
    result         = None      # (pred, conf, probs)
    captured_frame = None      # Frame figee pour la correction

    print("\nWebcam ouverte. Place un dechet devant la camera.")
    print("   [ESPACE] Classifier  |  [1-{}] Corriger  |  [q] Quitter\n".format(len(labels)))

    # --- MODIFICATION : Generer la liste des touches de correction proprement ---
    touches_valides = []
    for i in range(1, len(labels) + 1):
        touches_valides.append(ord(str(i)))

    while True:
        if state == "live":
            ret, frame = cap.read()
            if not ret:
                break
            display = draw_ui(frame.copy(), labels, state="live")
        else:
            display = draw_ui(captured_frame.copy(), labels, result=result, state="captured")

        cv2.imshow("Waste Classifier", display)

        # -- Apercu 224x224 -- ce que le modele voit exactement --
        preview_src = captured_frame if state == "captured" else frame
        preview = cv2.resize(preview_src, (224, 224))
        label_color = (0, 200, 100) if state == "live" else (0, 140, 255)
        preview = cv2.copyMakeBorder(preview, 20, 2, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(preview, "Vue modele 224x224", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, label_color, 1)
        cv2.imshow("Vue modele", preview)

        key = cv2.waitKey(1) & 0xFF

        # -- QUITTER --
        if key == ord("q"):
            break

        # -- CAPTURER & CLASSIFIER (Stabilise) --
        elif key == ord(" "):
            print("[ANALYSE] Stabilisation du résultat (Moyennage sur 5 frames)...")
            all_probs = []
            for _ in range(5):
                ret, f = cap.read()
                if not ret: break
                h, w = f.shape[:2]
                crop = f[h//2-112:h//2+112, w//2-112:w//2+112]
                tensor = preprocess(crop)
                probs = model.predict(tensor, verbose=0)[0]
                all_probs.append(probs)
                captured_frame = f.copy()

            if all_probs:
                avg_probs = np.mean(all_probs, axis=0)
                idx = np.argmax(avg_probs)
                pred, conf = labels[idx], float(avg_probs[idx])
                result = (pred, conf, avg_probs)
                state = "captured"

                if conf < CONFIDENCE_MIN:
                    print(f"[DOUTE] Confiance faible ({conf*100:.1f}%) -> {pred}. Merci de corriger.")
                else:
                    print(f"[PREDICTION] Predit : {pred}  ({conf*100:.1f}%)")

        # -- CORRECTION NUMERIQUE --
        elif state == "captured" and key in touches_valides:
            idx        = key - ord("1")
            true_label = labels[idx]
            pred, conf, _ = result

            if true_label == pred:
                print(f"[SUCCES] Confirme : {true_label}")
            else:
                path = save_correction(captured_frame, true_label)
                print(f"[SAUVEGARDE] Correction enregistree -> {path}  ({pred} -> {true_label})")

            state  = "live"
            result = None

        # -- VALIDER SANS CORRECTION --
        elif state == "captured" and key == ord("o"):
            print(f"[SUCCES] OK, la prediction '{result[0]}' etait correcte.")
            state  = "live"
            result = None


    cap.release()
    cv2.destroyAllWindows()
    print("[FIN] Fermeture.")


if __name__ == "__main__":
    run()