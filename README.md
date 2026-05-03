# MARJ - Classification de Déchets

MARJ est une application de classification de déchets en temps réel utilisant le deep learning (MobileNetV2) et TensorFlow/Keras.

## Fonctionnalités
- **Entraînement** : Script complet pour entraîner le modèle sur des données personnalisées.
- **Classification en direct** : Interface webcam pour classifier les objets.
- **Correction active** : Possibilité de corriger les erreurs de l'IA en direct pour affiner le modèle (Fine-tuning).

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation
1. **Entraînement** :
   ```bash
   python train.py
   ```
2. **Classification** :
   ```bash
   python classify.py
   ```

## Contrôles (Classification)
- `ESPACE` : Capturer et classifier.
- `1-4` : Corriger la classe si l'IA s'est trompée.
- `S` : Relancer un entraînement rapide sur les corrections.
- `Q` : Quitter.
