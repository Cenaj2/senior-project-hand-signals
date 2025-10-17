# train_nn.py
# Hybrid input: each tile is 3-channel (grayscale, edges, motion)
# - Live capture computes motion as absdiff from previous ROI frame
# - Saved training images are single ROIs; during dataset load motion channel = zeros
# - Model uses 9 branches (3x3 tiles) each input shape = (50,50,3)

import os
import cv2
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers, models, Input

# --- Optional: enable TF GPU memory growth so it doesn't allocate all memory ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

print("TensorFlow", tf.__version__)
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# ---------------- CONFIG ----------------
DATA_DIR = "gesture_data"
MODEL_FILE = "edge_multi_input_cnn.h5"
MODEL_TFLITE = "edge_multi_input_cnn_optimized.tflite"
LABEL_FILE = "labels.npy"

IMG_SIZE = 50           # tile size
NUM_INPUTS = 9          # 3x3
EPOCHS = 50
BATCH_SIZE = 16
DEBUG_SHOW = True       # comment out / set False for headless

# --- Skin calibration & detection settings ---
SKIN_PAD_YCRCB = np.array([15, 20])
SKIN_PAD_HSV = np.array([10, 40, 40])
MIN_HAND_AREA = 2500
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ---------- UTILS ----------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ---------- MODEL ----------
def cnn_branch(input_shape=(IMG_SIZE, IMG_SIZE, 3), branch_id=0):
    """
    One CNN branch per tile.
    NOTE: input_shape here is 3-channel (gray, edges, motion).
    """
    model = models.Sequential(name=f"cnn_branch_{branch_id}")
    # --- 10 convolutional layers (keeps compute reasonable) ---
    for i in range(10):
        filters = 16 * (i // 3 + 1)
        # first layer receives input_shape; others infer automatically
        model.add(
            layers.Conv2D(
                filters,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=input_shape if i == 0 else None,
                name=f"conv_{branch_id}_{i}",
            )
        )
        if i % 3 == 2:
            model.add(layers.MaxPooling2D((2, 2), name=f"pool_{branch_id}_{i}"))
    model.add(layers.Flatten(name=f"flatten_{branch_id}"))
    return model


def build_multi_input_cnn(num_classes):
    """
    Build a multi-input model with NUM_INPUTS branches,
    each expecting (IMG_SIZE, IMG_SIZE, 3).
    """
    inputs, branches = [], []
    for i in range(NUM_INPUTS):
        inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name=f"input_{i+1}")
        out = cnn_branch(input_shape=(IMG_SIZE, IMG_SIZE, 3), branch_id=i)(inp)
        inputs.append(inp)
        branches.append(out)
    merged = layers.Concatenate(name="concat_branches")(branches)
    dense = layers.Dense(128, activation="relu", name="dense_merge")(merged)
    output = layers.Dense(num_classes, activation="softmax", name="output")(dense)
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------- SKIN CALIBRATION ----------
def calibrate_skin(cap, num_frames=30, box_size=80):
    """
    Interactive skin calibration. Press 'c' with your hand in the green box.
    Returns dict of thresholds or None if aborted.
    """
    print("[INFO] Skin calibration: place hand inside box and press 'c'")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cx, cy = w // 2, h // 2
    half = box_size // 2
    samples_ycrcb, samples_hsv = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        disp = frame.copy()
        cv2.rectangle(disp, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 0), 2)
        cv2.putText(
            disp,
            "Place hand & press C",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibrate", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            print("[INFO] Capturing samples for calibration...")
            for _ in range(num_frames):
                ret, frame2 = cap.read()
                if not ret:
                    break
                box = frame2[cy - half : cy + half, cx - half : cx + half]
                if box.size == 0:
                    continue
                samples_ycrcb.append(cv2.cvtColor(box, cv2.COLOR_BGR2YCrCb).reshape(-1, 3))
                samples_hsv.append(cv2.cvtColor(box, cv2.COLOR_BGR2HSV).reshape(-1, 3))
            break
        elif key == ord("q"):
            cv2.destroyWindow("Calibrate")
            return None
    cv2.destroyWindow("Calibrate")

    if not samples_ycrcb:
        print("[ERROR] Calibration failed: No samples captured.")
        return None

    samples_ycrcb = np.vstack(samples_ycrcb)
    samples_hsv = np.vstack(samples_hsv)
    med_ycrcb, std_ycrcb = np.median(samples_ycrcb, 0), np.std(samples_ycrcb, 0)
    med_hsv, std_hsv = np.median(samples_hsv, 0), np.std(samples_hsv, 0)

    cr_med, cb_med = med_ycrcb[1], med_ycrcb[2]
    cr_std, cb_std = std_ycrcb[1], std_ycrcb[2]
    ycrcb_lower = np.array(
        [
            0,
            max(0, int(cr_med - SKIN_PAD_YCRCB[0] - cr_std)),
            max(0, int(cb_med - SKIN_PAD_YCRCB[1] - cb_std)),
        ],
        np.uint8,
    )
    ycrcb_upper = np.array(
        [
            255,
            min(255, int(cr_med + SKIN_PAD_YCRCB[0] + cr_std)),
            min(255, int(cb_med + SKIN_PAD_YCRCB[1] + cb_std)),
        ],
        np.uint8,
    )

    h_med, s_med, v_med = med_hsv
    hsv_lower = np.array(
        [
            max(0, int(h_med - SKIN_PAD_HSV[0] - std_hsv[0])),
            max(50, int(s_med - SKIN_PAD_HSV[1] - std_hsv[1])),
            max(30, int(v_med - SKIN_PAD_HSV[2] - std_hsv[2])),
        ],
        np.uint8,
    )
    hsv_upper = np.array(
        [min(179, int(h_med + SKIN_PAD_HSV[0] + std_hsv[0])), 255, 255], np.uint8
    )

    print("[INFO] Calibration complete.")
    return {
        "ycrcb_lower": ycrcb_lower,
        "ycrcb_upper": ycrcb_upper,
        "hsv_lower": hsv_lower,
        "hsv_upper": hsv_upper,
    }


# ---------- HAND SEGMENTATION + stacked channels ----------
def extract_hand_stack(frame, skin_thresh, bg_sub=None, prev_gray_roi=None):
    """
    Returns:
      stacked: (150,150,3) uint8 array with channels [gray, edges, motion]
      mask: full-frame mask (for debug)
      roi_resized: color ROI (150x150) for saving if needed
      new_prev_gray_roi: grayscale roi to use next frame as prev
    Motion channel = absdiff between current ROI gray and prev_gray_roi (if provided),
    else zeros.
    """
    if skin_thresh is None:
        # nothing calibrated, return blank
        blank = np.zeros((150,150,3), dtype=np.uint8)
        return blank, np.zeros(frame.shape[:2], dtype=np.uint8), np.zeros((150,150,3), dtype=np.uint8), None

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_y = cv2.inRange(ycrcb, skin_thresh["ycrcb_lower"], skin_thresh["ycrcb_upper"])
    mask_h = cv2.inRange(hsv, skin_thresh["hsv_lower"], skin_thresh["hsv_upper"])
    mask = cv2.bitwise_or(mask_y, mask_h)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)

    if bg_sub is not None:
        fg = bg_sub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, fg)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = frame
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        if cv2.contourArea(largest) >= MIN_HAND_AREA:
            x, y, w, h = cv2.boundingRect(hull)
            pad = int(0.2 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            roi = frame[y0:y1, x0:x1]

    roi_resized = cv2.resize(roi, (150, 150))
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 80, 180)

    if prev_gray_roi is not None:
        motion = cv2.absdiff(gray, prev_gray_roi)
    else:
        motion = np.zeros_like(gray)

    # stack channels (uint8) - later normalized to float when feeding model
    stacked = np.stack([gray, edges, motion], axis=-1)
    return stacked, mask, roi_resized, gray


def split_into_tiles_from_stack(stacked):
    """
    stacked: (150,150,3) uint8
    returns list of 9 tiles each shaped (50,50,3) normalized float32 [0,1]
    """
    tiles = []
    for yy in range(0, 150, 50):
        for xx in range(0, 150, 50):
            tile = stacked[yy:yy + 50, xx:xx + 50, :]  # (50,50,3)
            tile = tile.astype(np.float32) / 255.0
            tiles.append(tile)
    return tiles


# ---------- MAIN ----------
def main():
    ensure_dir(DATA_DIR)
    gesture = input("Enter gesture name: ").strip()
    gesture_path = os.path.join(DATA_DIR, gesture)
    ensure_dir(gesture_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # calibrate skin
    skin_thresh = calibrate_skin(cap)
    if skin_thresh is None:
        print("[ERROR] Calibration aborted.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # background subtractor for better FG mask
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    recording = False
    counter = 0
    prev_gray_roi = None  # used to compute motion channel during live capture

    print("[INFO] Press SPACE to start/stop recording, Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stacked, mask, roi, prev_gray_roi = extract_hand_stack(frame, skin_thresh, bg_sub, prev_gray_roi)
        disp = frame.copy()
        cv2.putText(
            disp,
            f"{gesture}: {counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if DEBUG_SHOW:
            # show mask, edges (first channel), motion (third channel), and ROI
            cv2.imshow("Mask", mask)
            cv2.imshow("Edges (ch1)", stacked[:, :, 1])
            cv2.imshow("Motion (ch2)", stacked[:, :, 2])
            cv2.imshow("ROI", roi)
            cv2.imshow("Camera", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
            print("[INFO] Recording..." if recording else "[INFO] Paused.")
        elif key == ord("q"):
            break

        if recording:
            # Save ROI (color 150x150). When training, we recompute stacked channels from this ROI.
            save_path = os.path.join(gesture_path, f"{gesture}_{counter}.png")
            cv2.imwrite(save_path, roi)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Captured", counter, "images for", gesture)

    # === TRAIN MODEL ===
    print("[INFO] Loading dataset and building training arrays...")
    t0 = time.time()
    X, y, label_map = [], [], {}
    idx = 0

    # load images and compute stacked channels (motion channel = zero for saved single-frame images)
    for folder in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(path):
            continue
        if folder not in label_map:
            label_map[folder] = idx
            idx += 1
        for file in sorted(os.listdir(path)):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # For saved images: compute stacked channels but motion = zeros
            # re-use extract_hand_stack but without bg_sub and prev_gray (we pass None)
            stacked, _, _, _ = extract_hand_stack(img, skin_thresh, bg_sub=None, prev_gray_roi=None)
            tiles = split_into_tiles_from_stack(stacked)  # returns 9 tiles of shape (50,50,3)
            if len(tiles) != NUM_INPUTS:
                # skip bad image
                continue
            X.append(tiles)
            y.append(label_map[folder])

    if not X:
        print("[ERROR] No training data found. Check if images were saved and calibration was successful.")
        return

    num_classes = len(label_map)
    total_samples = len(X)
    print(f"[INFO] Training model on {num_classes} gestures with {total_samples} samples total.")

    # --- Debug sample check ---
    print(f"[DEBUG] Class map: {label_map}")
    print(f"[DEBUG] Label distribution: {np.bincount(np.array(y))}")

    # convert to per-input numpy arrays
    inputs = [np.array([x[i] for x in X], dtype=np.float32) for i in range(NUM_INPUTS)]
    # verify shapes
    for i, arr in enumerate(inputs):
        print(f"[DEBUG] input[{i}] shape = {arr.shape}")
    print(f"[DEBUG] labels shape = {np.array(y).shape}")

    # build and train model
    model = build_multi_input_cnn(num_classes)

    t1 = time.time()
    print(f"[INFO] Dataset prepared in {(t1 - t0)*1000:.2f} ms")

    history = model.fit(
        inputs,
        np.array(y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    t2 = time.time()
    print(f"[INFO] Training finished in {(t2 - t1):.2f} seconds")

    # save model and labels
    ensure_dir(os.path.dirname(MODEL_FILE) or ".")
    model.save(MODEL_FILE)
    np.save(LABEL_FILE, label_map)
    print("[INFO] Saved model and label map.")

    # --- Convert to optimized TFLite (quantized) ---
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(MODEL_TFLITE, "wb") as f:
            f.write(tflite_model)
        print(f"[INFO] Optimized TFLite model saved as {MODEL_TFLITE}")
    except Exception as e:
        print("[WARN] TFLite conversion failed:", e)


if __name__ == "__main__":
    main()
