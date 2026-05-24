# Octobin - Waste Classification Prototype

Octobin is a real-time waste classification prototype using **MobileNetV2**, **TensorFlow/Keras**, and **OpenCV**.

The goal is to test a practical workflow: train a small classifier, run webcam inference, collect corrections, and improve the model.

## Features

- Training script with MobileNetV2 transfer learning.
- Webcam classification interface.
- Manual correction workflow for misclassified examples.
- Optional Flask app entry point.
- Training curves exported as an image.

## Requirements

Recommended Python version: **Python 3.10 or 3.11**.

TensorFlow support depends on your OS and Python version. Avoid Python 3.13 for this project unless TensorFlow support is confirmed for your environment.

```bash
pip install -r requirements.txt
```

## Expected dataset structure

```text
data/
  train/
    papier/
    plastique/
    verre/
    organique/
  val/
    papier/
    plastique/
    verre/
    organique/
```

If no dataset is available, the current script can generate small dummy images for testing the pipeline. These dummy images are only for smoke tests and should not be used to evaluate model quality.

## Usage

Train:

```bash
python train.py
```

Run webcam classification:

```bash
python classify.py
```

Run Flask app:

```bash
python app.py
```

## Controls

- `SPACE`: capture and classify
- `1-4`: correct the predicted class
- `S`: retrain on collected corrections, if supported by the current script version
- `Q`: quit

## Project status

Prototype. The model and UX are useful for experimentation, but the project still needs a stronger dataset, tests, evaluation metrics, and reproducible training configuration.
