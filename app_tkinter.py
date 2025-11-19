# app_tkinter.py
# GUI with tabs:
#   - Live Dashboard (webcam + spoof + emotion + rPPG + register + attendance)
#   - Attendance History
#   - User Management (delete templates)

import os
import time
import tkinter as tk
from tkinter import Label, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

import ttkbootstrap as tb
from ttkbootstrap.constants import *

# ==== Import logic & config from test_webcam.py ====
from test_webcam import (
    # models & loaders
    load_embedding_model,
    load_anti_spoof_model,
    load_emotion_model,
    load_registered_embeddings,

    # utilities / classes
    EMASmoother,
    EmotionSmoother,
    RPPG_CHROM,
    extract_face_mediapipe,
    get_forehead_roi_from_mesh,
    predict_identity,
    predict_p_live,
    predict_emotion_probs,
    log_attendance,
    preprocess_face_bgr,

    # constants
    COS_THRESHOLD,
    LIVE_THRESHOLD,
    EMA_ALPHA_SPOOF,
    EMA_ALPHA_FPS,
    EMO_WINDOW,
    EMO_LABELS,
    RPPG_WINDOW,
    RPPG_MIN_BPM,
    RPPG_MAX_BPM,
    RPPG_CONF_GOOD,
    UNCERTAIN_LOW,
    UNCERTAIN_HIGH,
    STABLE_FRAMES_FOR_ATTEND,
    IMG_SIZE_FACE,
    EMB_DIR,
    ATTENDANCE_CSV,

    # Mediapipe modules
    mp_face_detection,
    mp_face_mesh,
)

NUM_SAMPLES_REGISTER = 30

# ================= INIT MODELS & ENGINES =================
print("[INFO] Loading models...")
emb_model = load_embedding_model()
anti_model = load_anti_spoof_model()
emo_model = load_emotion_model()

# Load embeddings (if not present, leave empty)
try:
    reg_embeddings = load_registered_embeddings()
except (FileNotFoundError, RuntimeError) as e:
    print(f"[WARN] {e}")
    print("[INFO] No embeddings found yet. Creating empty registry.")
    os.makedirs(EMB_DIR, exist_ok=True)
    reg_embeddings = {}

# Smoothers
spoof_smoother = EMASmoother(EMA_ALPHA_SPOOF)
fps_smoother = EMASmoother(EMA_ALPHA_FPS)
emo_smoother = EmotionSmoother(window=EMO_WINDOW)

# rPPG engine
rppg_engine = RPPG_CHROM(
    window=RPPG_WINDOW,
    min_bpm=RPPG_MIN_BPM,
    max_bpm=RPPG_MAX_BPM,
)

# Mediapipe
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6,
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

print("[INFO] Models & engines loaded.")

# ================= GLOBAL STATE =================
seen_users = set()
stable_user = None
stable_count = 0

frame_idx = 0
current_emo_label = None  # Last emotion to log attendance

register_state = {
    "active": False,
    "user_id": None,
    "emb_list": [],
}

# ================= ROOT WINDOW + THEME =================
root = tb.Window(themename="cyborg")  # try "superhero", "cyborg", etc. if you like
root.title("Face Verification System (Dashboard)")
root.configure(bg="#222222")

# Optional header bar (makes it feel like a product)
header = tb.Frame(root, bootstyle="dark")
header.pack(fill="x")

tb.Label(
    header,
    text="Facial Recognition Attendance System",
    font=("Segoe UI", 18, "bold"),
    bootstyle="light",
).pack(side="left", padx=15, pady=10)

tb.Label(
    header,
    text="Face Verification • Anti-Spoofing • Emotion • rPPG",
    font=("Segoe UI", 11),
    bootstyle="secondary",
).pack(side="left", padx=10, pady=10)

# Notebook for tabs
notebook = tb.Notebook(root, bootstyle="primary")
notebook.pack(fill="both", expand=True)

# --- Tab 1: Live Dashboard ---
tab_live = tb.Frame(notebook, bootstyle="dark")
notebook.add(tab_live, text="Live Dashboard")

# --- Tab 2: Attendance History ---
tab_att = tb.Frame(notebook, bootstyle="dark")
notebook.add(tab_att, text="Attendance History")

# --- Tab 3: User Management ---
tab_users = tb.Frame(notebook, bootstyle="dark")
notebook.add(tab_users, text="User Management")

# ========== TAB 1: LIVE DASHBOARD LAYOUT ==========
# Layout:
#   - Left: video
#   - Right: status cards + register button
tab_live.columnconfigure(0, weight=3)
tab_live.columnconfigure(1, weight=1)
tab_live.rowconfigure(0, weight=1)
tab_live.rowconfigure(1, weight=0)
tab_live.rowconfigure(2, weight=0)
tab_live.rowconfigure(3, weight=0)

# Video area (left)
video_label = Label(tab_live, bg="#222222")
video_label.grid(
    row=0,
    column=0,
    rowspan=3,
    sticky="nsew",
    padx=10,
    pady=10,
)

# Right panel for text info
info_frame = tb.Frame(tab_live, bootstyle="dark")
info_frame.grid(
    row=0,
    column=1,
    sticky="nwe",
    padx=(0, 10),
    pady=10,
)
info_frame.columnconfigure(0, weight=1)

title_label = Label(
    info_frame,
    text="Live Dashboard",
    font=("Segoe UI", 16, "bold"),
    bg="#222222",
    fg="white",
)
title_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

id_label_ui = tb.Label(
    info_frame,
    text="ID: —",
    font=("Segoe UI", 13),
    bootstyle="inverse-dark",
    anchor="w",
    padding=10,
)
id_label_ui.grid(row=1, column=0, sticky="ew", pady=4)

spoof_label_ui = tb.Label(
    info_frame,
    text="Liveness: —",
    font=("Segoe UI", 13),
    bootstyle="inverse-secondary",
    anchor="w",
    padding=10,
)
spoof_label_ui.grid(row=2, column=0, sticky="ew", pady=4)

emo_label_ui = tb.Label(
    info_frame,
    text="Emotion: —",
    font=("Segoe UI", 13),
    bootstyle="inverse-info",
    anchor="w",
    padding=10,
)
emo_label_ui.grid(row=3, column=0, sticky="ew", pady=4)

bpm_label_ui = tb.Label(
    info_frame,
    text="rPPG: —",
    font=("Segoe UI", 13),
    bootstyle="inverse-secondary",
    anchor="w",
    padding=10,
)
bpm_label_ui.grid(row=4, column=0, sticky="ew", pady=4)

fps_label_ui = tb.Label(
    info_frame,
    text="FPS: —",
    font=("Segoe UI", 11),
    bootstyle="secondary",
    anchor="w",
)
fps_label_ui.grid(row=5, column=0, sticky="w", pady=(8, 0))

def on_register_id():
    """Enable multi-frame registration mode."""
    global register_state, reg_embeddings

    if register_state["active"]:
        messagebox.showinfo(
            "Register ID",
            f"Registering for '{register_state['user_id']}'. "
            "Wait until completed before registering others.",
        )
        return

    user_id = simpledialog.askstring(
        "Register ID",
        "Enter ID/name for new user:",
        parent=root,
    )
    if not user_id:
        return

    user_id = user_id.strip()
    if user_id in reg_embeddings:
        overwrite = messagebox.askyesno(
            "Duplicate ID",
            f"ID '{user_id}' already exists. Do you want to override the embedding?",
        )
        if not overwrite:
            return

    register_state["active"] = True
    register_state["user_id"] = user_id
    register_state["emb_list"] = []

    messagebox.showinfo(
        "Register ID",
        f"Start registration for '{user_id}'.\n"
        f"Look straight into the camera, keep your face REAL.\n"
        f"The system will automatically collect {NUM_SAMPLES_REGISTER} frames.",
    )

register_button = tb.Button(
    info_frame,
    text="Register New ID",
    bootstyle=SUCCESS,
    command=on_register_id,
)
register_button.grid(row=6, column=0, sticky="ew", pady=(12, 0))

status_label = Label(
    tab_live,
    text="Starting camera...",
    font=("Segoe UI", 11),
    bg="#222222",
    fg="#CCCCCC",
    anchor="w",
)
status_label.grid(
    row=3,
    column=0,
    columnspan=2,
    sticky="ew",
    padx=10,
    pady=(0, 8),
)

# ========== TAB 2: ATTENDANCE ==========
tab_att.rowconfigure(1, weight=1)
tab_att.columnconfigure(0, weight=1)

att_label = Label(
    tab_att,
    text="Attendance History (attendance.csv)",
    font=("Segoe UI", 13, "bold"),
    bg="#222222",
    fg="white",
)
att_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))

att_tree = tb.Treeview(
    tab_att,
    columns=("timestamp", "user_id", "bpm", "emotion"),
    show="headings",
    bootstyle="dark",
)
att_tree.heading("timestamp", text="Timestamp")
att_tree.heading("user_id", text="User ID")
att_tree.heading("bpm", text="BPM")
att_tree.heading("emotion", text="Emotion")
att_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))

def load_attendance():
    """read attendance.csv and load into Treeview."""
    att_tree.delete(*att_tree.get_children())
    if not os.path.exists(ATTENDANCE_CSV):
        return
    import csv

    with open(ATTENDANCE_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            # timestamp, user_id, bpm, emotion
            if len(row) >= 4:
                att_tree.insert("", "end", values=(row[0], row[1], row[2], row[3]))
            elif len(row) >= 2:
                # fallback if there is an old file
                att_tree.insert("", "end", values=(row[0], row[1], "", ""))

btn_refresh_att = tb.Button(
    tab_att,
    text="Refresh",
    bootstyle=INFO,
    command=load_attendance,
)
btn_refresh_att.grid(row=2, column=0, sticky="e", padx=10, pady=(0, 10))

# ========== TAB 3: USER MANAGEMENT ==========
tab_users.rowconfigure(1, weight=1)
tab_users.columnconfigure(0, weight=1)

users_label = Label(
    tab_users,
    text="Registered Users (embeddings/*.npy)",
    font=("Segoe UI", 13, "bold"),
    bg="#222222",
    fg="white",
)
users_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))

users_listbox = tk.Listbox(tab_users, height=10, bg="#2b2b2b", fg="white")
users_listbox.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=10,
    pady=(0, 8),
)

def refresh_users_list():
    """Load the user list from reg_embeddings into the Listbox."""
    users_listbox.delete(0, tk.END)
    for uid in sorted(reg_embeddings.keys()):
        users_listbox.insert(tk.END, uid)

def delete_selected_user():
    """Delete user: delete .npy file + delete from reg_embeddings."""
    selection = users_listbox.curselection()
    if not selection:
        messagebox.showwarning("Delete User", "Select a user first.")
        return
    idx = selection[0]
    uid = users_listbox.get(idx)

    confirm = messagebox.askyesno(
        "Delete User",
        f"Are you sure you want to delete '{uid}'?\n"
        f"File embeddings/{uid}.npy will be deleted.",
    )
    if not confirm:
        return

    # Delete file
    path = os.path.join(EMB_DIR, f"{uid}.npy")
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"[USER] Deleted file: {path}")
    except Exception as e:
        print("[ERROR] delete file:", e)

    # Delete from dict
    if uid in reg_embeddings:
        del reg_embeddings[uid]
        print(f"[USER] Removed from registry: {uid}")

    refresh_users_list()

btn_frame_users = tb.Frame(tab_users, bootstyle="dark")
btn_frame_users.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
btn_frame_users.columnconfigure(0, weight=1)
btn_frame_users.columnconfigure(1, weight=1)

btn_refresh_users = tb.Button(
    btn_frame_users,
    text="Refresh Users",
    bootstyle=INFO,
    command=refresh_users_list,
)
btn_refresh_users.grid(row=0, column=0, sticky="ew", padx=(0, 4))

btn_delete_user = tb.Button(
    btn_frame_users,
    text="Delete Selected",
    bootstyle=DANGER,
    command=delete_selected_user,
)
btn_delete_user.grid(row=0, column=1, sticky="ew", padx=(4, 0))

# Load list for the first time
refresh_users_list()

# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

def update_frame():
    global frame_idx, stable_user, stable_count, seen_users, register_state, current_emo_label

    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Failed to read camera frame.")
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    frame_idx += 1

    # 0) Face detection
    face_bgr, bbox = extract_face_mediapipe(display_frame, face_detection)

    if face_bgr is None:
        raw_fps = 1.0 / (time.time() - frame_start + 1e-9)
        fps = fps_smoother.update(raw_fps)
        cv2.putText(
            display_frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )

        if register_state["active"]:
            cv2.putText(
                display_frame,
                f"Register {register_state['user_id']}: No face",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )

        status_label.config(text="No face detected")

        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_label.img_tk = img_tk
        video_label.config(image=img_tk)

        # Update UI labels in "no face" case
        id_label_ui.config(text="ID: —", bootstyle="inverse-dark")
        spoof_label_ui.config(text="Liveness: —", bootstyle="inverse-secondary")
        emo_label_ui.config(text="Emotion: —")
        bpm_label_ui.config(text="rPPG: —")
        fps_label_ui.config(text=f"FPS: {fps:.1f}")

        root.after(10, update_frame)
        return

    x, y, w, h = bbox

    # 1) Anti-spoof CNN
    p_live_raw = predict_p_live(anti_model, face_bgr)
    p_live = spoof_smoother.update(p_live_raw)
    p_spoof = 1.0 - p_live

    # 2) rPPG + FaceMesh (UI + tie-break)
    rppg_text = "rPPG: N/A"
    rppg_color = (200, 200, 200)
    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    mesh_results = None
    if frame_idx % 2 == 0:
        mesh_results = face_mesh.process(frame_rgb)

    forehead_roi = get_forehead_roi_from_mesh(display_frame, mesh_results) if mesh_results else None
    roi_for_rppg = forehead_roi if forehead_roi is not None else face_bgr

    bpm, rppg_conf = rppg_engine.update(roi_for_rppg, time.time())

    rppg_good = False
    if bpm is not None and rppg_conf is not None:
        if (RPPG_MIN_BPM <= bpm <= RPPG_MAX_BPM) and (rppg_conf >= RPPG_CONF_GOOD):
            rppg_good = True

    if bpm is not None and rppg_conf is not None:
        rppg_text = f"rPPG: {bpm:.0f} bpm (c={rppg_conf:.2f})"
        if rppg_good:
            rppg_color = (0, 255, 0)      # good signal
        else:
            rppg_color = (0, 255, 255)    # weak / noise
    elif rppg_conf is not None:
        rppg_text = f"rPPG: N/A (c={rppg_conf:.2f})"
        rppg_color = (0, 255, 255)

    # 2b) Tie-break between CNN & rPPG
    is_live_cnn = (p_live >= LIVE_THRESHOLD)
    is_spoof_cnn = not is_live_cnn

    is_spoof = is_spoof_cnn  # default trust CNN

    if p_live <= UNCERTAIN_LOW:
        is_spoof = True                    # spoof
    elif p_live >= UNCERTAIN_HIGH:
        is_spoof = False                   # live
    else:
        if rppg_good:
            is_spoof = False
        else:
            is_spoof = is_spoof_cnn

    if is_spoof:
        bpm = None
        rppg_text = "rPPG: N/A"
        rppg_color = (0, 0, 255)

    # 3) Verification (REAL only)
    if not is_spoof:
        best_user, best_score = predict_identity(emb_model, reg_embeddings, face_bgr)
    else:
        best_user, best_score = None, 0.0

    # 4) Emotion
    emo_text = "N/A"
    if not is_spoof and (frame_idx % 3 == 0):
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

    # 5) Register mode (collect embeddings)
    if register_state["active"]:
        uid = register_state["user_id"]
        if not is_spoof:
            inp = preprocess_face_bgr(face_bgr, IMG_SIZE_FACE)
            emb = emb_model.predict(inp, verbose=0)[0]
            register_state["emb_list"].append(emb)

        collected = len(register_state["emb_list"])
        cv2.putText(
            display_frame,
            f"Register {uid}: {collected}/{NUM_SAMPLES_REGISTER}",
            (20, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        if collected >= NUM_SAMPLES_REGISTER:
            emb_array = np.stack(register_state["emb_list"], axis=0)
            emb_mean = emb_array.mean(axis=0).astype("float32")

            os.makedirs(EMB_DIR, exist_ok=True)
            save_path = os.path.join(EMB_DIR, f"{uid}.npy")
            np.save(save_path, emb_mean)
            reg_embeddings[uid] = emb_mean

            messagebox.showinfo(
                "Register ID",
                f"ID registration successful '{uid}'.\n"
                f"Saved average embedding at:\n{save_path}",
            )
            print(f"[REGISTER] Saved mean embedding for {uid} -> {save_path}")
            print("[REGISTER] Norm:", np.linalg.norm(emb_mean))

            register_state["active"] = False
            register_state["user_id"] = None
            register_state["emb_list"] = []
            refresh_users_list()

    # 6) Attendance (stable user)
    if (not is_spoof) and (best_user is not None) and (best_score >= COS_THRESHOLD):
        if best_user == stable_user:
            stable_count += 1
        else:
            stable_user = best_user
            stable_count = 1
    else:
        stable_user = None
        stable_count = 0

    if (
        (stable_user is not None)
        and (stable_count >= STABLE_FRAMES_FOR_ATTEND)
        and (stable_user not in seen_users)
    ):
        seen_users.add(stable_user)
        log_attendance(stable_user, bpm, current_emo_label)
        load_attendance()  # update tab Attendance

    is_verified = (stable_user is not None) and (stable_user in seen_users)

    # 7) draw ID / SPOOF
    if is_spoof:
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

    cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

    # 8) FPS
    raw_fps = 1.0 / (time.time() - frame_start + 1e-9)
    fps = fps_smoother.update(raw_fps)

    # 9) Overlay text on video
    cv2.putText(
        display_frame,
        id_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        id_color,
        2,
    )
    cv2.putText(
        display_frame,
        f"LiveProb={p_live:.2f} | TH={LIVE_THRESHOLD}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        1,
    )
    cv2.putText(
        display_frame,
        f"Emotion: {emo_text}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        display_frame,
        f"CosTH={COS_THRESHOLD}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        display_frame,
        rppg_text,
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        rppg_color,
        1,
    )
    cv2.putText(
        display_frame,
        f"FPS: {fps:.1f}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1,
    )

    status_label.config(text=f"{id_text} | {emo_text} | {rppg_text}")

    # ---- Update the dashboard labels (right side) ----
    id_label_ui.config(text=f"ID: {id_text}")

    if is_spoof:
        spoof_label_ui.config(text="Liveness: 🛑 SPOOF", bootstyle="inverse-danger")
    else:
        if best_user is not None and best_score >= COS_THRESHOLD:
            if is_verified:
                spoof_label_ui.config(
                    text="Liveness: ✅ LIVE (Verified)",
                    bootstyle="inverse-success",
                )
            else:
                spoof_label_ui.config(text="Liveness: ✅ LIVE", bootstyle="inverse-success")
        else:
            spoof_label_ui.config(text="Liveness: —", bootstyle="inverse-secondary")

    emo_label_ui.config(text=f"Emotion: {emo_text}")
    bpm_label_ui.config(text=rppg_text)
    fps_label_ui.config(text=f"FPS: {fps:.1f}")

    # 10) Show Tkinter video
    img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    video_label.img_tk = img_tk
    video_label.config(image=img_tk)

    root.after(10, update_frame)

def on_closing():
    if cap.isOpened():
        cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Load attendance
load_attendance()

update_frame()
root.mainloop()
