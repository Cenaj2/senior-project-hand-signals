import cv2, os, time, numpy as np, tensorflow as tf

MODEL_FILE = "edge_multi_input_cnn.h5"
LABEL_FILE = "labels.npy"
IMG_SIZE = 50
NUM_INPUTS = 9
DELAY_MS = 200        # adjustable delay between predictions
SHOW_IM = False         # comment or set False for headless
DEBUG_SHOW = False      # shows edge tiles and contour detection

# --- Preprocess frame ---
def preprocess_frame(frame):
    """Extracts 9 edge tiles focused on the hand area."""
    # Convert to YCrCb to isolate skin regions
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77])
    upper = np.array([255, 173, 127])
    mask = cv2.inRange(ycrcb, lower, upper)

    # Morphological cleanup
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find largest contour (likely the hand)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        hand_roi = frame[y:y+h, x:x+w]
    else:
        hand_roi = frame

    # Resize region to standard 150x150
    roi_resized = cv2.resize(hand_roi, (150,150))

    # Edge detection on cropped ROI
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)

    # Split into 9 tiles (3x3)
    tiles = []
    for yy in range(0,150,50):
        for xx in range(0,150,50):
            tile = edges[yy:yy+50, xx:xx+50]
            tile = tile.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0
            tiles.append(np.expand_dims(tile, 0))

    return tiles, edges, mask, roi_resized

def main():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_FILE):
        print("[ERROR] Model or label file missing.")
        return
    model = tf.keras.models.load_model(MODEL_FILE)
    label_map = np.load(LABEL_FILE, allow_pickle=True).item()
    rev_map = {v:k for k,v in label_map.items()}

    cap = cv2.VideoCapture(0)
    print("[INFO] Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        tiles, edges, mask, roi_resized = preprocess_frame(frame)
        preds = model.predict([t for t in tiles], verbose=0)
        pred_class = np.argmax(preds)
        label = rev_map.get(pred_class, "Unknown")
        conf = np.max(preds)*100
        print(f"Gesture: {label} ({conf:.1f}%)")

        if SHOW_IM:
            display = frame.copy()
            cv2.putText(display, f"{label} ({conf:.1f}%)", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # Draw detected hand contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(display, contours, -1, (0,255,0), 2)
            cv2.imshow("Recognition", display)

            # Optional debug windows
            if DEBUG_SHOW:
                cv2.imshow("Edges", edges)
                cv2.imshow("SkinMask", mask)
                # combine 9 tiles horizontally for quick view
                tiles_vis = [np.hstack([edges[y:y+50, x:x+50] for x in range(0,150,50)]) for y in range(0,150,50)]
                combined = np.vstack(tiles_vis)
                cv2.imshow("9 Input Tiles", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        time.sleep(DELAY_MS / 1000.0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
