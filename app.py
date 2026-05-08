import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- CONFIG --------------------------------------------------------------------
MODEL_PATH      = "waste_model.keras"
LABELS_PATH     = "labels.txt"
CORRECTIONS_DIR = "corrections"   # Dossier pour les corrections manuelles
UPLOAD_FOLDER   = "static/uploads"
IMG_SIZE        = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creation des repertoires necessaires
os.makedirs(CORRECTIONS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- CHARGEMENT ----------------------------------------------------------------
def charger_labels():
    """Charge les labels depuis le disque."""
    if not os.path.exists(LABELS_PATH):
        print(f"[ERREUR] Labels introuvables : {LABELS_PATH}")
        return []
    
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        # On filtre les lignes vides et on strip les espaces
        labels = [l.strip() for l in f if l.strip()]
    return labels

def charger_modele_et_labels():
    """Charge le modele TensorFlow et les labels."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERREUR] Modele introuvable : {MODEL_PATH}")
        print("  Lance d'abord : python train.py")
        return None, []
    
    modele = tf.keras.models.load_model(MODEL_PATH)
    labels = charger_labels()
            
    print(f"[SUCCES] Modele charge -- {len(labels)} classes : {labels}")
    return modele, labels

# Initialisation du modele au demarrage
modele, labels = charger_modele_et_labels()


# --- PRE-TRAITEMENT ------------------------------------------------------------
def pre_traitement(chemin_img):
    """Preparation de l'image conforme a MobileNetV2."""
    img = cv2.imread(chemin_img)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return np.expand_dims(preprocess_input(img), axis=0)


# --- ROUTES FLASK --------------------------------------------------------------

@app.route('/')
def index():
    """Affiche l'interface web principale."""
    global labels
    labels = charger_labels() # Rechargement dynamique pour eviter les problemes
    return render_template('index.html', labels=labels)


@app.route('/classify', methods=['POST'])
def classifier():
    """Point d'entree pour la classification d'une image envoyee par l'UI."""
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image recue'}), 400
    
    fichier = request.files['image']
    if fichier.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    # Sauvegarde du fichier avec un horodatage unique
    nom_fichier = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fichier.filename}")
    chemin_complet = os.path.join(app.config['UPLOAD_FOLDER'], nom_fichier)
    fichier.save(chemin_complet)

    if modele is None:
        return jsonify({'error': 'Le modele n\'est pas pret'}), 500

    # Analyse de l'image
    tensor = pre_traitement(chemin_complet)
    probs  = modele.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    
    pred   = labels[idx]
    conf   = float(probs[idx])
    
    print(f"[ANALYSE] {nom_fichier} -> Predit : {pred} ({conf*100:.1f}%)")

    return jsonify({
        'prediction': pred,
        'confidence': conf,
        'image_url': f"/{chemin_complet}"
    })


@app.route('/correct', methods=['POST'])
def corriger():
    """Enregistre une correction utilisateur dans le dossier corrections/."""
    data = request.json
    url_img = data.get('image_url')
    vrai_label = data.get('true_label')

    if not url_img or not vrai_label:
        return jsonify({'error': 'Donnees incompletes'}), 400

    # Nettoyage du chemin
    chemin_local = url_img.lstrip('/')
    if not os.path.exists(chemin_local):
        return jsonify({'error': 'Image source introuvable sur le disque'}), 404

    # Rangement de l'image pour le futur fine-tuning
    dossier = os.path.join(CORRECTIONS_DIR, vrai_label)
    os.makedirs(dossier, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nouveau_chemin = os.path.join(dossier, f"{ts}.jpg")
    
    # On enregistre une copie dans le dossier corrections
    img = cv2.imread(chemin_local)
    cv2.imwrite(nouveau_chemin, img)
    
    print(f"[SAUVEGARDE] Correction enregistree pour {vrai_label} -> {nouveau_chemin}")

    return jsonify({'success': True})


@app.route('/stats')
def statistiques():
    """Retourne le nombre d'images de correction par classe."""
    stats_data = {}
    for lbl in labels:
        dossier = os.path.join(CORRECTIONS_DIR, lbl)
        if os.path.exists(dossier):
            stats_data[lbl] = len(os.listdir(dossier))
        else:
            stats_data[lbl] = 0
    return jsonify(stats_data)


if __name__ == '__main__':
    # Demarrage de l'application
    print("[INFO] Lancement du serveur MARJ-juju sur le port 5000...")
    app.run(debug=True, port=5000)
