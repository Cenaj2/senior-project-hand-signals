import os
import cv2
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers, models, Input

# ---------------- CONFIG ----------------
DATA_DIR = "gesture_data"
MODEL_FILE = "edge_multi_input_cnn.h5"
MODEL_TFLITE = "edge_multi_input_cnn_optimized.tflite"
LABEL_FILE = "labels.npy"

IMG_SIZE = 50
NUM_INPUTS = 9
EPOCHS = 10
BATCH_SIZE = 16
DEBUG_SHOW = True # ← comment out to run headless

# --- Skin calibration & detection settings ---
SKIN_PAD_YCRCB = np.array([15, 20])
SKIN_PAD_HSV = np.array([10, 40, 40])
MIN_HAND_AREA = 2500
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- Finger Detection Tuning ---
# FIX APPLIED HERE: Increased epsilon for more aggressive smoothing to fix the cv2.error.
CONTOUR_SIMPLIFY_EPSILON = 0.03 
# Minimum depth for a convexity defect to be counted as a valley/finger separation
MIN_DEFECT_DEPTH = 20.0 


# ---------- UTILS ----------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ---------- MODEL ----------
def cnn_branch(input_shape=(IMG_SIZE, IMG_SIZE, 1), branch_id=0):
    """One CNN branch per tile."""
    model = models.Sequential(name=f"cnn_branch_{branch_id}")
    for i in range(10): # 10 convolutional layers
        filters = 16 * (i // 3 + 1)
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
    """Builds a 9-input CNN for 9 image tiles."""
    inputs, branches = [], []
    for i in range(NUM_INPUTS):
        inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name=f"input_{i+1}")
        out = cnn_branch(branch_id=i)(inp)
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
    print("[INFO] Skin calibration: place hand inside box and press 'c'")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
            print("[INFO] Capturing samples...")
            for _ in range(num_frames):
                ret, frame2 = cap.read()
                if not ret:
                    break
                box = frame2[cy - half : cy + half, cx - half : cx + half]
                samples_ycrcb.append(cv2.cvtColor(box, cv2.COLOR_BGR2YCrCb).reshape(-1, 3))
                samples_hsv.append(cv2.cvtColor(box, cv2.COLOR_BGR2HSV).reshape(-1, 3))
            break
        elif key == ord("q"):
            cv2.destroyWindow("Calibrate")
            return None # Returns None on abort
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


# ---------- HAND SEGMENTATION AND FINGER TRACKING (Modified) ----------
def extract_hand_edges(frame, skin_thresh, bg_sub=None):
    """Improved contour-based hand isolation + edge extraction, including finger detection."""
    
    if skin_thresh is None:
        return np.zeros((150, 150), dtype=np.uint8), np.zeros(frame.shape[:2], dtype=np.uint8), np.zeros((150, 150, 3), dtype=np.uint8)

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
    finger_count = 0
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        
        if contour_area >= MIN_HAND_AREA:
            
            # 1. Simplify the contour aggressively to prevent the 'not monotonous' error
            epsilon = CONTOUR_SIMPLIFY_EPSILON * cv2.arcLength(largest, True)
            approx_contour = cv2.approxPolyDP(largest, epsilon, True)
            
            # Check if the simplified contour is complex enough (must have at least 3 points)
            if approx_contour.shape[0] >= 3:
                
                # Use the simplified contour for the hull points for a more robust bounding box
                hull_points = cv2.convexHull(approx_contour)
                x, y, w, h = cv2.boundingRect(hull_points)
                
                # Draw a bounding box for visualization on the original frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                pad = int(0.2 * max(w, h))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                roi = frame[y0:y1, x0:x1]
                
                # 2. Finger Counting using Convexity Defects
                # Get hull indices from the SIMPLIFIED contour
                hull_indices = cv2.convexHull(approx_contour, returnPoints=False)
                
                # Must have at least 3 points AND 4 indices for a valid defect calculation
                if hull_indices.shape[0] > 3:
                    # Defects must be calculated on the SIMPLIFIED contour and its hull indices
                    defects = cv2.convexityDefects(approx_contour, hull_indices)
                    
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            # Use approx_contour instead of largest for points
                            farthest = tuple(approx_contour[f][0]) 
                            
                            # Distance between farthest point and convex hull (depth)
                            depth = d / 256.0 
                            
                            if depth > MIN_DEFECT_DEPTH: 
                                finger_count += 1
                                # Draw circle at the valley (between fingers)
                                cv2.circle(frame, farthest, 5, [0, 0, 255], -1)
                    
                    # Finger count logic (number of defects + 1 for thumb/base)
                    if finger_count > 0:
                        finger_count += 1 
    
    # Add finger count to the display
    if finger_count > 0:
        cv2.putText(
            frame, 
            f"FINGERS: {finger_count}", 
            (frame.shape[1] - 150, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )

    # The rest of the function remains the same for model input preparation
    roi_resized = cv2.resize(roi, (150, 150))
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 80, 180)
    
    return edges, mask, roi_resized


def split_into_tiles(edges):
    """Split 150x150 edge map into 9 tiles."""
    tiles = []
    for yy in range(0, 150, 50):
        for xx in range(0, 150, 50):
            tile = edges[yy : yy + 50, xx : xx + 50]
            tile = tile.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0
            tiles.append(tile)
    return tiles


# ---------- MAIN (Remains mostly the same) ----------
def main():
    ensure_dir(DATA_DIR)
    gesture = input("Enter gesture name: ").strip()
    gesture_path = os.path.join(DATA_DIR, gesture)
    ensure_dir(gesture_path)

    cap = cv2.VideoCapture(0)
    skin_thresh = calibrate_skin(cap)

    # --- THE FIX: Check if calibration was successful ---
    if skin_thresh is None:
        print("[ERROR] Skin calibration aborted. Exiting application.")
        cap.release()
        cv2.destroyAllWindows()
        return
    # ----------------------------------------------------

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )

    recording = False
    counter = 0
    print("[INFO] Press SPACE to start/stop recording, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # extract_hand_edges now also modifies the 'frame' with finger tracking data
        edges, mask, roi = extract_hand_edges(frame, skin_thresh, bg_sub)
        
        # We copy the potentially modified 'frame' to 'disp' for display
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
            cv2.imshow("Mask", mask)
            cv2.imshow("Edges", edges)
            cv2.imshow("ROI", roi)
            cv2.imshow("Camera", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
            print("[INFO] Recording..." if recording else "[INFO] Paused.")
        elif key == ord("q"):
            break
        if recording:
            cv2.imwrite(os.path.join(gesture_path, f"{gesture}_{counter}.png"), roi) 
            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Captured", counter, "images.")

    # === TRAIN MODEL ===
    print("[INFO] Loading dataset...")
    t0 = time.time()
    X, y, label_map = [], [], {}
    idx = 0
    for folder in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(path):
            continue
        if folder not in label_map:
            label_map[folder] = idx
            idx += 1
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # When extracting for training, the finger drawing is just ignored, 
            # and the model still trains on the edges.
            edges, _, _ = extract_hand_edges(img, skin_thresh) 
            tiles = split_into_tiles(edges)
            X.append(tiles)
            y.append(label_map[folder])

    if not X:
        print("[ERROR] No training data found. Check if images were saved and calibration was successful.")
        return

    num_classes = len(label_map)
    print(
        f"[INFO] Training model on {num_classes} gestures with {len(X)} samples total."
    )

    # --- Debug sample check ---
    print(f"Class map: {label_map}")
    print(f"Label distribution: {np.bincount(np.array(y))}")

    inputs = [np.array([x[i] for x in X]) for i in range(NUM_INPUTS)]

    model = build_multi_input_cnn(num_classes)

    t1 = time.time()
    print(f"[INFO] Dataset loaded in {(t1 - t0)*1000:.2f} ms")

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

    ensure_dir(os.path.dirname(MODEL_FILE) or ".")
    model.save(MODEL_FILE)
    np.save(LABEL_FILE, label_map)
    print("[INFO] Saved model and label map.")

    # --- Convert to TFLite ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Optimized TFLite model saved as {MODEL_TFLITE}")


if __name__ == "__main__":
    main()