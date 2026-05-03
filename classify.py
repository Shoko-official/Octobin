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

# --- CONFIG --------------------------------------------------------------------
MODEL_PATH      = "waste_model.keras"
LABELS_PATH     = "labels.txt"
CORRECTIONS_DIR = "corrections"   # Images mal classees sauvegardees ici
IMG_SIZE        = (224, 224)
CONFIDENCE_MIN  = 0.70          # Seuil de confiance minimal (alors 70 c haut oui mais softmax peut etre trop confiant)
FINETUNE_LR     = 1e-5
FINETUNE_EPOCHS = 3

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
    
    labels = []
    with open(LABELS_PATH) as f:
        # --- MODIFICATION : Boucle classique au lieu d'une liste en une ligne ---
        for ligne in f.readlines():
            labels.append(ligne.strip())
            
    print(f"[SUCCES] Modele charge -- {len(labels)} classes : {labels}")
    return model, labels


# --- PRE-TRAITEMENT ------------------------------------------------------------
def preprocess(frame: np.ndarray) -> np.ndarray:
    """Redimensionne et normalise une frame pour MobileNetV2."""
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


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


# --- FINE-TUNING SUR LES CORRECTIONS ------------------------------------------
def finetune_on_corrections(model, labels: list) -> None:
    """Re-entraine le modele sur les images corrigees accumulees."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if not os.path.exists(CORRECTIONS_DIR):
        print("[ATTENTION] Aucune correction enregistree.")
        return

    # --- MODIFICATION : Boucle classique pour compter les images ---
    total = 0
    for _, _, fichiers in os.walk(CORRECTIONS_DIR):
        if fichiers:
            total += len(fichiers)
            
    if total == 0:
        print("[ATTENTION] Dossier corrections/ vide.")
        return

    print(f"\n[CONFIGURATION] Fine-tuning sur {total} correction(s)...")

    gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    data = gen.flow_from_directory(
        CORRECTIONS_DIR,
        target_size=IMG_SIZE,
        batch_size=max(1, min(8, total)),
        class_mode="categorical",
        classes=labels,
    )

    if data.num_classes != len(labels):
        print("[ATTENTION] Toutes les classes ne sont pas representees dans corrections/. "
              "Ajoute au moins 1 image par classe ou patiente.")
        return

    # Degele les dernieres couches
    model.trainable = True
    for layer in model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(data, epochs=FINETUNE_EPOCHS, verbose=1)
    model.save(MODEL_PATH)
    print(f"[SUCCES] Modele mis a jour -> {MODEL_PATH}")


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
    cv2.putText(frame, "[s] Re-train  [q] Quitter",
                (w - 190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

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
    print("   [ESPACE] Classifier  |  [1-{}] Corriger  |  [s] Re-train  |  [q] Quitter\n".format(len(labels)))

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

        # -- CAPTURER & CLASSIFIER --
        elif key == ord(" "):
            ret, frame = cap.read()
            if ret:
                captured_frame = frame.copy()
                result = predict(model, captured_frame, labels)
                pred, conf, _ = result
                state = "captured"

                if conf < CONFIDENCE_MIN:
                    print(f"[DOUTE] Confiance faible ({conf*100:.1f}%) -> {pred}. Merci de corriger.")
                else:
                    print(f"[PREDICTION] Predit : {pred}  (confiance : {conf*100:.1f}%)")

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

        # -- RE-TRAIN --
        elif key == ord("s"):
            finetune_on_corrections(model, labels)

    cap.release()
    cv2.destroyAllWindows()
    print("[FIN] Fermeture.")


if __name__ == "__main__":
    run()