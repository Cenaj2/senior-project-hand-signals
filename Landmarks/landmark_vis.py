import os
import cv2
import json
import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_FILE = "landmark_detector.h5"   # or your .tflite if you use TF Lite
IMG_SIZE = 288                           # full image size (for visualization)
NUM_LANDMARKS = 7                        # number of keypoints
NUM_INPUTS = 9                           # tiles per image (multi-input)
TILE_SIZE = 96                           # tile size used in training

# ---------------- HELPERS ----------------
def split_into_tiles(img):
    """Split image into 9 tiles just like during training."""
    tiles = []
    h, w = img.shape[:2]
    step_y, step_x = h // 3, w // 3
    for yy in range(0, h, step_y):
        for xx in range(0, w, step_x):
            tile = img[yy:yy+step_y, xx:xx+step_x]
            tile = cv2.resize(tile, (TILE_SIZE, TILE_SIZE))
            tiles.append(tile)
    return tiles

# ---------------- MAIN ----------------
folder = input("Enter gesture folder: ").strip()
json_path = os.path.join(folder, "landmarks.json")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Annotation file not found: {json_path}")

print(f"[INFO] Loading model: {MODEL_FILE}")
model = tf.keras.models.load_model(MODEL_FILE)

with open(json_path, "r") as f:
    data = json.load(f)

samples = data.get("samples", data)

for i, sample in enumerate(samples):
    img_path = os.path.join(folder, sample.get("file", sample.get("image", "")))
    if not os.path.exists(img_path):
        print(f"[WARN] Missing {img_path}, skipping.")
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Preprocess
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    tiles = split_into_tiles(img)
    inputs = [np.expand_dims(tile / 255.0, 0) for tile in tiles]

    # Predict
    preds = model.predict(inputs, verbose=0)[0]
    pts = preds.reshape(-1, 2) * IMG_SIZE  # denormalize to pixel coordinates

    # Draw keypoints and connections
    for (x, y) in pts.astype(int):
        cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

    # Optional: connect landmarks
    if len(pts) == 7:
        palm, thumb, index, middle, ring, pinky, wrist = pts.astype(int)
        for a, b in [(palm, wrist), (palm, thumb), (palm, index),
                     (palm, middle), (palm, ring), (palm, pinky)]:
            cv2.line(img, tuple(a), tuple(b), (255, 0, 0), 2)

    cv2.imshow(f"Sample {i+1}/{len(samples)}", img)
    print(f"[INFO] Displaying {img_path} - press any key for next, 'q' to quit.")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(f"Sample {i+1}/{len(samples)}")
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("[INFO] Visualization complete.")
