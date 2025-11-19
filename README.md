# 📘 Facial Recognition Attendance System with Emotion & Liveness Detection

A complete end-to-end intelligent attendance system integrating:

🔐 Face Verification (Hybrid CNN + Transformer embeddings)

🛡️ Anti-Spoofing / Liveness Detection (Texture CNN + rPPG-CHROM)

😊 Emotion Recognition (Real-time facial expression model)

🎥 Real-Time Video Pipeline using Mediapipe Face Detection & FaceMesh

🖥️ Tkinter GUI for dashboard, user management, and attendance history

This project was developed for COS30082 – Applied Machine Learning, Swinburne University of Technology.

## 🚀 1. System Overview

This system performs secure, real-time face recognition and provides a complete attendance pipeline with additional features:

### ✔ Face Verification (Open-Set)

- Hybrid EfficientNetB4 + CBAM + TinyViT backbone

- L2-normalized 512-D embeddings

- Cosine similarity for identity matching

- Threshold-based open-set verification

### ✔ Anti-Spoofing (Liveness Detection)

Two complementary methods were integrated:

**1. CNN Anti-Spoof Model**

Detects printed photos, screens, and replay attacks using texture & reflectance cues.

**2. rPPG-CHROM (Remote Photoplethysmography)**
Extracts subtle skin-color oscillations across frames using FaceMesh.
Fake faces have no natural blood-flow → low liveness score.

### ✔ Emotion Detection

- Trained on FER2013 + RAF-DB

- Seven emotions: angry, disgust, fear, happy, neutral, sad, surprise

- Rolling-window smoothing for stable output

### ✔ Tkinter GUI

Includes:

- Live Dashboard (webcam, identity, spoof, emotion, rPPG, FPS)

- Attendance History

- User Management (delete templates, view registered users)

## 📁 2. Repository Structure

```bash
project-root/
│
├── app_tkinter.py                # Main GUI application
├── test_webcam.py                # Real-time inference pipeline
├── model_layers.py               # Custom hybrid backbone
│
├── face-verification-model/      # Embedding model (.keras)
├── anti-spoof-model/             # Spoof classifier model
├── emotion-model/                # Emotion classification model
│
├── 3-Models-Code/                # Training notebooks
│
├── embeddings/                   # Saved user face embeddings
├── attendance_logs/              # Auto-generated CSV logs
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ 3. Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/dannguyen0309/face-recognition-system.git
cd face-recognition-system
```

### 2️⃣ Create & activate a virtual environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### ▶️ 4. Running the Application

**Start the Tkinter GUI:**

```bash
python app_tkinter.py
```

Features include:

- Real-time face recognition
- Automatic attendance logging
- Spoof detection (CNN + rPPG)
- Emotion prediction
- New user registration

## 👤 Author

Nguyen Ngoc Lam Dan
Swinburne University of Technology – Vietnam
COS30082 Applied Machine Learning

## License

[MIT](https://choosealicense.com/licenses/mit/)
