"""
train.py -- Entrainement du CNN de classification de dechets
------------------------------------------------------------
Utilise MobileNetV2 (transfer learning) pour classer les dechets.

Structure de dossiers attendue :
  data/
    train/
      papier/     -> images .jpg/.png
      plastique/
      verre/
      organique/
    val/          (meme structure, optionnel -- cree auto si absent)

Usage :
  python train.py
"""

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --- CONFIG --------------------------------------------------------------------
DATA_DIR    = "data"
TRAIN_DIR   = os.path.join(DATA_DIR, "train") 
VAL_DIR     = os.path.join(DATA_DIR, "val") 
MODEL_PATH  = "waste_model.keras"  # Nom du fichier final contenant notre IA entraînée
LABELS_PATH = "labels.txt" 

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16 #lecture par paquet de 16
EPOCHS      = 15 #Le nombre de fois que l'IA va parcourir l'intégralité du dataset

# Classes de dechets (doivent correspondre aux noms de sous-dossiers !!!!)
CLASSES = ["papier", "plastique", "verre", "organique"]

# --- CREATION D'UN DATASET FACTICE SI VIDE -------------------------------------
def create_sample_data():
    """Cree des images colorees factices si le dossier data/ est vide."""
    print("[ATTENTION] Dossier data/ vide -- creation d'images factices pour tester...")
    colors = {
        "papier":    (220, 200, 150),
        "plastique": (0, 120, 255),
        "verre":     (0, 200, 200),
        "organique": (50, 150, 50),
    }
    for split in ["train", "val"]:
        n = 40 if split == "train" else 10 # 40 images pour s'entraîner, 10 pour valider
        for cls, color in colors.items():
            folder = os.path.join(DATA_DIR, split, cls)
            os.makedirs(folder, exist_ok=True) # Crée le dossier s'il n'existe pas
            for i in range(n):
                img = np.full((224, 224, 3), color, dtype=np.uint8)
                # Ajoute du bruit pour eviter des images identiques (que se soit différent un tout petit peu)
                noise = np.random.randint(0, 40, (224, 224, 3), dtype=np.uint8)
                img = np.clip(img.astype(int) + noise - 20, 0, 255).astype(np.uint8)
                from PIL import Image
                Image.fromarray(img).save(os.path.join(folder, f"{cls}_{i}.jpg"))
    print("[SUCCES] Donnees factices creees.")


def auto_split_val():
    """Si val/ n'existe pas, prend 20% des images de train/ pour la validation."""
    if os.path.exists(VAL_DIR):
        return ## Si le dossier val existe déjà on fait rien
    print("[INFO] Dossier val/ absent -- split automatique 80/20 depuis train/...")
    
    #Parcour tous les dossiers de déchets dans train
    for cls in os.listdir(TRAIN_DIR):
        src = os.path.join(TRAIN_DIR, cls)
        if not os.path.isdir(src):
            continue
        
        dst = os.path.join(VAL_DIR, cls)
        os.makedirs(dst, exist_ok=True)
        
        # Récupère toutes les images du dossier courant
        imgs = []
        for f in os.listdir(src):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                imgs.append(f)
        
        # Mélange les images au hasard pour ne pas prendre toujours les mêmes
        np.random.shuffle(imgs)
        
        # Calcule 20% du nombre total d'images
        limite = max(1, len(imgs) // 5)
        #on les déplace vers val
        for img in imgs[:limite]:
            shutil.move(os.path.join(src, img), os.path.join(dst, img))
            
    print("[SUCCES] Split val/ effectue.")


# --- VERIFICATION DES DONNEES --------------------------------------------------
def check_data():
    if not os.path.exists(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
        create_sample_data()
    auto_split_val()


# --- CONSTRUCTION DU MODELE ----------------------------------------------------
def build_model(num_classes: int) -> Model:
    """MobileNetV2 + tete de classification fine-tunee."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3), #224*224, 3 canaux rgb
        include_top=False, # On retire la couche finale d'origine (qui classait 1000 objets aléatoires)
        weights="imagenet" #on prend un modèle semi-entrainé
    )
    # Gele le backbone pour l'entrainement initial
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# --- ENTRAINEMENT --------------------------------------------------------------
def train():
    check_data()

    # Generateurs d'images
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical"
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical"
    )

    # --- MODIFICATION ICI : Boucles classiques au lieu de comprehensions ---
    # 1. Inverser le dictionnaire (clé <-> valeur)
    label_map = {}
    for k, v in train_data.class_indices.items():
        label_map[v] = k

    # 2. Creer la liste des labels dans le bon ordre
    labels = []
    for i in range(len(label_map)):
        labels.append(label_map[i])
    # ------------------------------------------------------------------------

    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(labels))
    print(f"[INFO] Classes detectees : {labels}")

    num_classes = len(labels)
    model = build_model(num_classes)
    model.summary()

    # Phase 1 : tete seule
    print("\n[DEBUT] Phase 1 -- Entrainement de la tete de classification...")
    history = model.fit(
        train_data, validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
        ]
    )

    # Phase 2 : fine-tuning des 30 dernieres couches
    print("\n[CONFIGURATION] Phase 2 -- Fine-tuning...")
    model.layers[1].trainable = True  # base = layers[1]
    for layer in model.layers[1].layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Recuperation de l'historique de la phase 2
    history2 = model.fit(
        train_data, validation_data=val_data,
        epochs=8,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ]
    )

    model.save(MODEL_PATH)
    print(f"\n[SUCCES] Modele sauvegarde -> {MODEL_PATH}")

    # --- Combinaison des historiques pour le graphique ---
    acc = history.history["accuracy"] + history2.history["accuracy"]
    val_acc = history.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history.history["loss"] + history2.history["loss"]
    val_loss = history.history["val_loss"] + history2.history["val_loss"]
    
    # Point de separation entre Phase 1 et Phase 2
    initial_epochs = len(history.history["accuracy"])

    # Courbe d'apprentissage
    plt.figure(figsize=(10, 4))
    
    # Graphique de precision (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train")
    plt.plot(val_acc, label="Val")
    plt.axvline(x=initial_epochs - 1, color='r', linestyle='--', label='Debut Fine-Tuning')
    plt.title("Precision")
    plt.legend()
    
    # Graphique de perte (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train")
    plt.plot(val_loss, label="Val")
    plt.axvline(x=initial_epochs - 1, color='r', linestyle='--', label='Debut Fine-Tuning')
    plt.title("Perte")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("[INFO] Courbes sauvegardees -> training_curves.png")


if __name__ == "__main__":
    train()