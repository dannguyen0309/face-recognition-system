# ===================== test_webcam.py (verify + anti-spoof + emotion + rPPG-CHROM + FaceMesh + stable attendance) =====================
import os
import time
import csv
import cv2
import numpy as np
import tensorflow as tf
import keras
from collections import deque
from tensorflow.keras.applications.efficientnet import preprocess_input
import mediapipe as mp

from model_layers import (
    TransformerBlock,
    avg_fn,
    max_fn,
    sp_out_shape,
    l2_fn,
    l2_out_shape,
)

# Mediapipe modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh      = mp.solutions.face_mesh

# ====================== CONFIG ======================
# Model paths
MODEL_PATH_FACE = r"./face-verification-model/emb_hybrid_b4_supervised.keras"
MODEL_PATH_ANTI = r"./anti-spoof-model/anti_spoof_from_hybrid.keras"
MODEL_PATH_EMO  = r"./emotion-model/emotion_from_hybrid_final.keras"

# Embeddings & attendance
EMB_DIR = r"./embeddings"
ATTENDANCE_CSV = "attendance.csv"

# Input sizes
IMG_SIZE_FACE = (224, 224)
IMG_SIZE_AUX  = (160, 160)

# Thresholds
COS_THRESHOLD  = 0.90        # face verification
LIVE_THRESHOLD = 0.60        # p(LIVE) < TH => spoof

EMO_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Smoothing params
EMA_ALPHA_SPOOF = 0.6        # 0.5–0.8 recommended
EMA_ALPHA_FPS   = 0.9
EMO_WINDOW      = 15         # frames for emotion smoothing

# rPPG params
RPPG_WINDOW     = 150        # frames for CHROM (~5s nếu 30fps)
RPPG_MIN_BPM    = 45.0
RPPG_MAX_BPM    = 180.0

# rPPG support (tie-break + quality)
RPPG_CONF_GOOD      = 0.12   # conf >= 0.12 consider as good signal
UNCERTAIN_LOW       = 0.45   # p_live < 0.45 ->  spoof 
UNCERTAIN_HIGH      = 0.80   # p_live > 0.80 -> live  

# Attendance stability: need multiple N-frame that detect correct user
STABLE_FRAMES_FOR_ATTEND = 20   
# ====================================================


# ================= EMA SMOOTHER =====================
class EMASmoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = float(new_value)
        else:
            self.value = self.alpha * self.value + (1.0 - self.alpha) * float(new_value)
        return self.value


# ========== EMOTION SMOOTHER (majority vote) ==========
class EmotionSmoother:
    def __init__(self, window=15):
        self.queue = deque(maxlen=window)

    def update(self, emo_idx):
        self.queue.append(int(emo_idx))
        if not self.queue:
            return None
        vals, counts = np.unique(list(self.queue), return_counts=True)
        return int(vals[np.argmax(counts)])


# ========== rPPG CHROM IMPLEMENTATION ==========
class RPPG_CHROM:
    """
    Remote PPG using CHROM algorithm (de Haan 2013).
    """
    def __init__(self, window=150, min_bpm=45.0, max_bpm=180.0):
        self.window = window
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.rgb_hist = deque(maxlen=window)
        self.t_hist   = deque(maxlen=window)

    def _extract_roi_rgb_means(self, roi_bgr):
        """
        Calculate mean RGB in ROI (cropped).
        """
        h, w, _ = roi_bgr.shape
        if h < 10 or w < 10:
            return None

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_rgb = roi_rgb.astype(np.float32) / 255.0
        r_mean = float(np.mean(roi_rgb[:, :, 0]))
        g_mean = float(np.mean(roi_rgb[:, :, 1]))
        b_mean = float(np.mean(roi_rgb[:, :, 2]))
        return r_mean, g_mean, b_mean

    def _chrom_signal(self, rgb_arr):
        R = rgb_arr[:, 0]
        G = rgb_arr[:, 1]
        B = rgb_arr[:, 2]

        # normalize by mean
        Rn = (R / (np.mean(R) + 1e-8)) - 1.0
        Gn = (G / (np.mean(G) + 1e-8)) - 1.0
        Bn = (B / (np.mean(B) + 1e-8)) - 1.0

        X = 3.0 * Rn - 2.0 * Gn
        Y = 1.5 * Rn + Gn - 1.5 * Bn

        alpha = np.std(X) / (np.std(Y) + 1e-8)
        S = X - alpha * Y
        return S

    def update(self, roi_bgr, t_now):
        """
        Call each frame with ROI (forehead) has cropped
        """
        roi_means = self._extract_roi_rgb_means(roi_bgr)
        if roi_means is None:
            return None, None

        self.rgb_hist.append(roi_means)
        self.t_hist.append(t_now)

        T = len(self.rgb_hist)
        if T < 64:
            return None, None

        rgb_arr = np.array(self.rgb_hist)
        ts = np.array(self.t_hist)

        duration = ts[-1] - ts[0]
        if duration <= 0:
            return None, None
        fs = (T - 1) / duration  # Hz

        sig = self._chrom_signal(rgb_arr)
        sig = sig - np.mean(sig)
        sig = sig * np.hanning(len(sig))

        freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        fft_mag = np.abs(np.fft.rfft(sig))

        f_min = self.min_bpm / 60.0
        f_max = self.max_bpm / 60.0
        band = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(band):
            return None, None

        freqs_band = freqs[band]
        mag_band   = fft_mag[band]

        idx_peak = np.argmax(mag_band)
        peak_freq = freqs_band[idx_peak]
        bpm = peak_freq * 60.0

        peak_amp = mag_band[idx_peak]
        conf = float(peak_amp / (np.sum(mag_band) + 1e-8))

        if bpm < self.min_bpm or bpm > self.max_bpm:
            return None, conf

        return float(bpm), conf


# ==================== UTILS ===========================
def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def preprocess_face_bgr(frame_bgr, img_size):
    """Resize + EfficientNet preprocess -> (1, H, W, 3)."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, img_size)
    img_resized = img_resized.astype("float32")
    img_resized = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_batch


def build_custom_objects():
    return {
        "TransformerBlock": TransformerBlock,
        "avg_fn": avg_fn,
        "max_fn": max_fn,
        "sp_out_shape": sp_out_shape,
        "l2_fn": l2_fn,
        "l2_out_shape": l2_out_shape,
    }


def load_keras_model(path):
    print(f"[INFO] TF: {tf.__version__} | Keras: {keras.__version__}")
    keras.config.enable_unsafe_deserialization()
    custom_objects = build_custom_objects()

    print(f"[INFO] Loading model from: {path}")
    model = keras.models.load_model(
        path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )
    print(f"[INFO] Model loaded: {model.name}")
    print("[INFO] Input shape:", model.input_shape,
          "-> Output shape:", model.output_shape)
    return model


def load_embedding_model():
    return load_keras_model(MODEL_PATH_FACE)


def load_anti_spoof_model():
    return load_keras_model(MODEL_PATH_ANTI)


def load_emotion_model():
    return load_keras_model(MODEL_PATH_EMO)


def load_registered_embeddings():
    """Load *.npy in EMB_DIR -> {user_id: emb_vector}."""
    if not os.path.isdir(EMB_DIR):
        raise FileNotFoundError(f"EMB_DIR not found: {EMB_DIR}")

    emb_dict = {}
    for fname in os.listdir(EMB_DIR):
        if not fname.lower().endswith(".npy"):
            continue
        user_id = os.path.splitext(fname)[0]
        path = os.path.join(EMB_DIR, fname)
        emb = np.load(path)
        emb_dict[user_id] = emb
        print(f"[INFO] Loaded {user_id} from {path}, norm={np.linalg.norm(emb):.3f}")

    if not emb_dict:
        raise RuntimeError(
            f"No .npy embeddings found in {EMB_DIR}. Run register_face.py first."
        )
    return emb_dict


def extract_face_mediapipe(frame_bgr, face_detection):
    """Detect 1 face, return (face_bgr, (x,y,w,h)) or (None, None)."""
    h, w, _ = frame_bgr.shape
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if not results.detections:
        return None, None

    det = max(results.detections, key=lambda d: d.score[0])
    bbox = det.location_data.relative_bounding_box

    x_min = max(int(bbox.xmin * w), 0)
    y_min = max(int(bbox.ymin * h), 0)
    x_max = min(int((bbox.xmin + bbox.width) * w), w - 1)
    y_max = min(int((bbox.ymin + bbox.height) * h), h - 1)

    if x_max <= x_min or y_max <= y_min:
        return None, None

    face = frame_bgr[y_min:y_max, x_min:x_max]
    return face, (x_min, y_min, x_max - x_min, y_max - y_min)


def get_forehead_roi_from_mesh(frame_bgr, mesh_results):
    """
    Using FaceMesh to take ROI forehead in full frame. return None if failed
    """
    if not mesh_results or not mesh_results.multi_face_landmarks:
        return None

    lm = mesh_results.multi_face_landmarks[0].landmark
    H, W, _ = frame_bgr.shape

    # Một số landmark quanh trán / trên lông mày
    idxs = [10, 338, 297, 67]  # mid forehead + left/right
    xs = [int(lm[i].x * W) for i in idxs]
    ys = [int(lm[i].y * H) for i in idxs]

    x1 = max(min(xs) - 5, 0)
    x2 = min(max(xs) + 5, W - 1)
    y1 = max(min(ys) - 20, 0)  # mở rộng lên trên
    y2 = min(max(ys) + 10, H - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame_bgr[y1:y2, x1:x2].copy()
    return roi


def predict_identity(emb_model, reg_embeddings, face_bgr):
    """Return (best_user, best_score)."""
    inp_face = preprocess_face_bgr(face_bgr, IMG_SIZE_FACE)
    emb = emb_model.predict(inp_face, verbose=0)[0]

    best_user = None
    best_score = -1.0
    for user_id, ref_emb in reg_embeddings.items():
        score = cosine_sim(ref_emb, emb)
        if score > best_score:
            best_score = score
            best_user = user_id
    return best_user, best_score


def predict_p_live(anti_model, face_bgr):
    """Return p(LIVE) from anti-spoof model."""
    inp_face = preprocess_face_bgr(face_bgr, IMG_SIZE_AUX)
    prob = anti_model.predict(inp_face, verbose=0)[0][0]
    return float(prob)


def predict_emotion_probs(emo_model, face_bgr):
    """Return logits/probs vector."""
    inp_face = preprocess_face_bgr(face_bgr, IMG_SIZE_AUX)
    probs = emo_model.predict(inp_face, verbose=0)[0]
    return probs


def log_attendance(user_id, bpm=None, emotion=None):
    """
    Append attendance record to CSV.
    Format: timestamp, user_id, bpm, emotion
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "user_id", "bpm", "emotion"])
        bpm_str = f"{bpm:.1f}" if bpm is not None else ""
        emo_str = emotion if emotion is not None else ""
        w.writerow([now, user_id, bpm_str, emo_str])
    print(f"[ATTENDANCE] VERIFIED {user_id} @ {now} | BPM={bpm_str} | EMO={emo_str}")



# ========================= MAIN =========================
def main():
    # Load models
    emb_model = load_embedding_model()
    anti_model = load_anti_spoof_model()
    emo_model  = load_emotion_model()
    reg_embeddings = load_registered_embeddings()

    # Attendance control
    seen_users = set()         # user has been written in CSV
    stable_user = None         # stable user
    stable_count = 0           # Number of consecutive frames seen with the same user

    # Smoothers & rPPG
    spoof_smoother = EMASmoother(EMA_ALPHA_SPOOF)
    fps_smoother   = EMASmoother(EMA_ALPHA_FPS)
    emo_smoother   = EmotionSmoother(window=EMO_WINDOW)
    rppg_engine    = RPPG_CHROM(window=RPPG_WINDOW,
                                min_bpm=RPPG_MIN_BPM,
                                max_bpm=RPPG_MAX_BPM)

    current_emo_label = None   # last emotion (string) to log attendance
    # Frame counter (to reduce the frequency of some tasks)
    frame_idx = 0

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    ) as face_detection, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        print("[INFO] Press 'q' to quit.")

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame.")
                break

            frame_idx += 1

            # ========== 0) Face detection ==========
            face_bgr, bbox = extract_face_mediapipe(frame, face_detection)
            if face_bgr is None:
                cv2.putText(
                    frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
                )
                raw_fps = 1.0 / (time.time() - frame_start + 1e-9)
                fps = fps_smoother.update(raw_fps)
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
                )
                cv2.imshow("Face System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            x, y, w, h = bbox

            # ========== 1) Anti-spoof CNN (main layer) ==========
            p_live_raw = predict_p_live(anti_model, face_bgr)
            p_live = spoof_smoother.update(p_live_raw)
            p_spoof = 1.0 - p_live
            is_spoof_cnn = (p_live < LIVE_THRESHOLD)   # SPOOF using CNN

            # ========== 2) rPPG CHROM + FaceMesh (support UI + tie-break) ==========
            rppg_text = "rPPG: N/A"
            rppg_color = (200, 200, 200)   # default grey
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mesh_results = None
            if frame_idx % 2 == 0:   # FaceMesh each 2 frame
                mesh_results = face_mesh.process(frame_rgb)

            forehead_roi = get_forehead_roi_from_mesh(frame, mesh_results) if mesh_results else None
            roi_for_rppg = forehead_roi if forehead_roi is not None else face_bgr

            bpm, rppg_conf = rppg_engine.update(roi_for_rppg, time.time())

            # rPPG Quality Assessment
            rppg_good = False
            if bpm is not None and rppg_conf is not None:
                if (RPPG_MIN_BPM <= bpm <= RPPG_MAX_BPM) and (rppg_conf >= RPPG_CONF_GOOD):
                    rppg_good = True

            # Text + color for UI
            if bpm is not None and rppg_conf is not None:
                rppg_text = f"rPPG: {bpm:.0f} bpm (c={rppg_conf:.2f})"
                if rppg_good:
                    rppg_color = (0, 255, 0)      # green: good signal
                else:
                    rppg_color = (0, 255, 255)    # yellow: weak signal / noise
            elif rppg_conf is not None:
                rppg_text = f"rPPG: N/A (c={rppg_conf:.2f})"
                rppg_color = (0, 255, 255)

            # ========== 2b) Tie-break logic between CNN & rPPG ==========
            # CNN base decision
            is_live_cnn = (p_live >= LIVE_THRESHOLD)
            is_spoof_cnn = not is_live_cnn

            # Default trust CNN
            is_spoof = is_spoof_cnn

            # If CNN is very sure -> trust CNN completely, leave rPPG
            if p_live <= UNCERTAIN_LOW:
                is_spoof = True
            elif p_live >= UNCERTAIN_HIGH:
                is_spoof = False
            else:
                # "Uncertain" zone -> for rPPG tie-break
                if rppg_good:
                    # rPPG good => leaning towards LIVE
                    is_spoof = False
                else:
                   # rPPG is not good => keep CNN decision
                    is_spoof = is_spoof_cnn
            
            # After the final decision is_spoof
            if is_spoof:
                # Don't show BPM anymore if you have concluded spoof
                bpm = None
                rppg_text = "rPPG: N/A"
                rppg_color = (0, 0, 255)   


            # ========== 3) Face verification (only final is REAL) ==========
            if not is_spoof:
                best_user, best_score = predict_identity(
                    emb_model, reg_embeddings, face_bgr
                )
            else:
                best_user, best_score = None, 0.0

            # ========== 4) Emotion (only REAL) + smoothing ==========
            emo_text = "N/A"
            if not is_spoof and (frame_idx % 3 == 0):    #
                probs = predict_emotion_probs(emo_model, face_bgr)
                idx_raw = int(np.argmax(probs))
                idx_smooth = emo_smoother.update(idx_raw)
                final_idx = idx_smooth if idx_smooth is not None else idx_raw
                emo_label = EMO_LABELS[final_idx]
                emo_conf = float(probs[final_idx])
                emo_text = f"{emo_label} ({emo_conf:.2f})"
                current_emo_label = emo_label          
            elif not is_spoof and emo_smoother.queue:
                last_idx = emo_smoother.queue[-1]
                emo_label = EMO_LABELS[last_idx]
                emo_text = f"{emo_label} (hold)"
                current_emo_label = emo_label         
            else:
                # If spoof or no emotion yet, can let current_emo_label keep old value
                pass


            # ========== 5) Attendance stability logic (use final is_spoof) ==========
            if (not is_spoof) and (best_user is not None) and (best_score >= COS_THRESHOLD):
                if best_user == stable_user:
                    stable_count += 1
                else:
                    stable_user = best_user
                    stable_count = 1
            else:
                stable_user = None
                stable_count = 0

            if (stable_user is not None and
                stable_count >= STABLE_FRAMES_FOR_ATTEND and
                stable_user not in seen_users):
                seen_users.add(stable_user)
                log_attendance(stable_user, bpm, current_emo_label)

            is_verified = (stable_user is not None) and (stable_user in seen_users)

            # ========== 6) Draw color frame according to final is_spoof & verified ==========
            if is_spoof:
                # Distinguish: spoof due to CNN or rPPG reject (for debugging)
                if is_spoof_cnn:
                    id_text = f"SPOOF_CNN ({p_spoof:.2f})"
                else:
                    id_text = f"SPOOF_rPPG ({p_spoof:.2f})"
                id_color = (0, 0, 255)
                box_color = (0, 0, 255)
            else:
                if best_user is not None and best_score >= COS_THRESHOLD:
                    if is_verified:
                        id_text = f"{best_user} ({best_score:.3f}) [VERIFIED]"
                        id_color = (255, 0, 0)
                        box_color = (255, 0, 0)
                    else:
                        id_text = f"{best_user} ({best_score:.3f})"
                        id_color = (0, 255, 0)
                        box_color = (0, 255, 0)
                else:
                    id_text = f"UNKNOWN ({best_score:.3f})"
                    id_color = (0, 255, 255)
                    box_color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # ========== 7) FPS smoothing ==========
            raw_fps = 1.0 / (time.time() - frame_start + 1e-9)
            fps = fps_smoother.update(raw_fps)

            # ========== 8) Overlay text ==========
            cv2.putText(
                frame, id_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, id_color, 2
            )
            cv2.putText(
                frame, f"LiveProb={p_live:.2f} | TH={LIVE_THRESHOLD}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1
            )
            cv2.putText(
                frame, f"Emotion: {emo_text}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"CosTH={COS_THRESHOLD}", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            cv2.putText(
                frame, rppg_text, (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rppg_color, 1
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
            )

            cv2.imshow("Face System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
