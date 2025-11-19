# Facial Recognition with Emotion and Liveness

This project is an attendance system that performs:

- Face verification (using a hybrid CNN/Transformer embedding model).
- Anti-spoofing / liveness detection (CNN-based + rPPG-CHROM).
- Emotion recognition from facial expressions.
- Real-time GUI using Tkinter.

## Main Scripts

- `app_tkinter.py` – Main GUI application (dashboard, attendance, user management).
- `test_webcam.py` – Core webcam pipeline (face detection, verification, spoof, emotion, rPPG).
- `model_layers.py` – Custom model layers and hybrid backbone components.

## Models

- `face-verification-model/`
- `anti-spoof-model/`
- `emotion-model/`

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```
