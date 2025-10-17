import cv2
import numpy as np
import tensorflow as tf
import time
import os

# ---------------- CONFIG ----------------
MODEL_TFLITE = "edge_multi_input_cnn_optimized.tflite"
LABEL_FILE = "labels.npy"
IMG_SIZE = 50
NUM_INPUTS = 9
DELAY_MS = 100
DEBUG_SHOW = True
MIN_HAND_AREA = 2500
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- Load labels ---
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError("No label file found! Run training first.")
label_map = np.load(LABEL_FILE, allow_pickle=True).item()
labels = {v: k for k, v in label_map.items()}

# --- Load TensorFlow Lite model ---
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        cv2.putText(disp, "Place hand & press C", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Calibrate", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            print("[INFO] Capturing samples...")
            for _ in range(num_frames):
                ret, frame2 = cap.read()
                if not ret:
                    break
                box = frame2[cy - half:cy + half, cx - half:cx + half]
                samples_ycrcb.append(cv2.cvtColor(box, cv2.COLOR_BGR2YCrCb).reshape(-1, 3))
                samples_hsv.append(cv2.cvtColor(box, cv2.COLOR_BGR2HSV).reshape(-1, 3))
            break
        elif key == ord("q"):
            cv2.destroyWindow("Calibrate")
            return None
    cv2.destroyWindow("Calibrate")

    samples_ycrcb = np.vstack(samples_ycrcb)
    samples_hsv = np.vstack(samples_hsv)
    med_ycrcb, std_ycrcb = np.median(samples_ycrcb, 0), np.std(samples_ycrcb, 0)
    med_hsv, std_hsv = np.median(samples_hsv, 0), np.std(samples_hsv, 0)

    cr_med, cb_med = med_ycrcb[1], med_ycrcb[2]
    cr_std, cb_std = std_ycrcb[1], std_ycrcb[2]
    ycrcb_lower = np.array(
        [0,
         max(0, int(cr_med - 15 - cr_std)),
         max(0, int(cb_med - 20 - cb_std))],
        np.uint8,
    )
    ycrcb_upper = np.array(
        [255,
         min(255, int(cr_med + 15 + cr_std)),
         min(255, int(cb_med + 20 + cb_std))],
        np.uint8,
    )

    h_med, s_med, v_med = med_hsv
    hsv_lower = np.array(
        [max(0, int(h_med - 10 - std_hsv[0])),
         max(50, int(s_med - 40 - std_hsv[1])),
         max(30, int(v_med - 40 - std_hsv[2]))],
        np.uint8,
    )
    hsv_upper = np.array(
        [min(179, int(h_med + 10 + std_hsv[0])), 255, 255],
        np.uint8,
    )

    print("[INFO] Calibration complete.")
    return {
        "ycrcb_lower": ycrcb_lower,
        "ycrcb_upper": ycrcb_upper,
        "hsv_lower": hsv_lower,
        "hsv_upper": hsv_upper,
    }

# ---------- HAND SEGMENTATION ----------
def extract_hand_stack(frame, skin_thresh, bg_sub=None):
    """Return 3-channel stack: grayscale, edges, and motion."""
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
        if cv2.contourArea(largest) >= MIN_HAND_AREA:
            x, y, w, h = cv2.boundingRect(largest)
            pad = int(0.2 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            roi = frame[y0:y1, x0:x1]

    roi_resized = cv2.resize(roi, (150, 150))
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 80, 180)

    # --- Motion detection ---
    motion = np.zeros_like(gray)
    if hasattr(extract_hand_stack, "prev_gray"):
        motion = cv2.absdiff(gray, extract_hand_stack.prev_gray)
    extract_hand_stack.prev_gray = gray.copy()

    # Stack into 3-channel hybrid
    stacked = np.stack([gray, edges, motion], axis=-1)
    return stacked, mask, roi_resized

# ---------- TILE SPLITTING ----------
def split_into_tiles(stacked):
    tiles = []
    for yy in range(0, 150, 50):
        for xx in range(0, 150, 50):
            tile = stacked[yy:yy + 50, xx:xx + 50]
            tile = tile.astype(np.float32) / 255.0
            tiles.append(tile)
    return tiles

# ---------- PREDICTION ----------
def predict_gesture(tiles):
    start_time = time.time()
    for i in range(NUM_INPUTS):
        interpreter.set_tensor(input_details[i]['index'],
                               tiles[i][np.newaxis, ...].astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    runtime_ms = (time.time() - start_time) * 1000
    idx = int(np.argmax(output))
    conf = float(output[idx])
    return labels.get(idx, "Unknown"), conf, runtime_ms

# ---------- MAIN LOOP ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    skin_thresh = calibrate_skin(cap)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    print("[INFO] Starting recognition. Press Q to quit.")
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (time.time() - prev_time) * 1000 < DELAY_MS:
            continue
        prev_time = time.time()

        stacked, mask, roi = extract_hand_stack(frame, skin_thresh, bg_sub)
        tiles = split_into_tiles(stacked)
        gesture, conf, runtime_ms = predict_gesture(tiles)

        print(f"Detected: {gesture:<15} | Confidence: {conf*100:5.1f}% | Inference: {runtime_ms:6.1f} ms")

        # --- Replace these with your custom actions ---
        # if gesture == "click" and conf > 0.9:  do_click()
        # elif gesture == "up" and conf > 0.9:   move_cursor_up()
        # ---------------------------------------------

        if DEBUG_SHOW:
            disp = frame.copy()
            cv2.putText(disp, f"{gesture} ({conf*100:.1f}%) {runtime_ms:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Mask", mask)
            cv2.imshow("Stacked Gray+Edges+Motion",
                       stacked.astype(np.uint8)[:, :, 0])
            cv2.imshow("Recognize", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recognition stopped.")

if __name__ == "__main__":
    main()
