# train_landmark_multi_input.py
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.losses import Huber
import time

# ---------------- CONFIG ----------------
DATA_DIR = "gesture_data"     # folder containing subfolders, each with images + aug_annotations.json
MODEL_FILE = "landmark_detector.h5"
MODEL_TFLITE = "landmark_multi_input_optimized.tflite"

IMG_SIZE = 96
TILE_FULL = 288            # full crop size that we split into 3x3 tiles -> 288 -> 9 tiles of 96
NUM_INPUTS = 9             # 9 tiles per image
NUM_LANDMARKS = 7          # 7 keypoints (x,y)
EPOCHS = 40
BATCH_SIZE = 16
VERBOSE = 1

# ---------------- HELPERS ----------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_into_tiles(img):
    """
    Splits a square image (expected TILE_FULL x TILE_FULL) into 9 tiles of IMG_SIZE.
    If image is not exactly TILE_FULL, it's resized first to TILE_FULL x TILE_FULL.
    Returns list of 9 tiles (H,W,3) as uint8 arrays.
    """
    img = cv2.resize(img, (TILE_FULL, TILE_FULL))
    tiles = []
    step = TILE_FULL // 3
    for yy in range(0, TILE_FULL, step):
        for xx in range(0, TILE_FULL, step):
            tile = img[yy:yy+step, xx:xx+step]
            tile = cv2.resize(tile, (IMG_SIZE, IMG_SIZE))
            tiles.append(tile)
    return tiles  # length 9

def load_json_annotations(path):
    """Load JSON that contains {"samples":[{ "file": "...", "landmarks":[ [x,y], ... ] }, ... ]} or a flat list."""
    with open(path, "r") as f:
        data = json.load(f)
    # Support both {"samples":[...]} and [... list ...]
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure in " + path)

def normalize_landmarks(pts, w, h):
    """If pts appear to be pixel coords (>1), normalize to 0..1, else assume already normalized."""
    pts = np.array(pts, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((NUM_LANDMARKS*2,), dtype=np.float32)
    # pts shape (7,2)
    if np.any(pts > 1.5):  # assume pixel coords if any coordinate > 1.5
        pts[:,0] = pts[:,0] / float(w)
        pts[:,1] = pts[:,1] / float(h)
    pts = np.clip(pts, 0.0, 1.0)
    return pts.flatten()  # length 14

# ---------------- DATA LOADING ----------------
def load_data():
    """
    Walks DATA_DIR subfolders. Looks for aug_annotations.json inside each folder.
    Each annotation entry: {"file":"img.png","points":[[x,y],...7points...] }
    Returns:
      X: numpy array of shape (N, 9, IMG_SIZE, IMG_SIZE, 3)
      y: numpy array of shape (N, 14) normalized coords
    """
    X_samples = []
    y_samples = []

    for folder in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        json_path_options = [
            os.path.join(folder_path, "aug_annotations.json"),
            os.path.join(folder_path, "landmarks.json"),
            os.path.join(folder_path, "landmarks.json"),
            os.path.join(folder_path, "aug_annotations.json")
        ]
        json_path = None
        for p in json_path_options:
            if os.path.exists(p):
                json_path = p
                break
        if json_path is None:
            # try any .json in folder
            for f in os.listdir(folder_path):
                if f.lower().endswith(".json"):
                    json_path = os.path.join(folder_path, f)
                    break
        if json_path is None:
            print(f"[WARN] no json annotation found in {folder_path}, skipping")
            continue

        samples = load_json_annotations(json_path)
        for entry in samples:
            fname = entry.get("file") or entry.get("image") or entry.get("img")
            pts = entry.get("points") or entry.get("landmarks") or entry.get("keypoints")
            if fname is None or pts is None:
                continue
            img_path = os.path.join(folder_path, fname)
            if not os.path.exists(img_path):
                # maybe filename includes path; try raw
                if os.path.exists(fname):
                    img_path = fname
                else:
                    # skip missing images
                    # print(f"[WARN] missing image {img_path}")
                    continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            norm = normalize_landmarks(pts, w, h)  # length 14

            tiles = split_into_tiles(img)  # list length 9 of (IMG_SIZE,IMG_SIZE,3)
            X_samples.append(tiles)
            y_samples.append(norm)

    if len(X_samples) == 0:
        return np.zeros((0, NUM_INPUTS, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), np.zeros((0, NUM_LANDMARKS*2), dtype=np.float32)

    X = np.array(X_samples, dtype=np.uint8)  # shape (N,9,H,W,3)
    y = np.array(y_samples, dtype=np.float32)  # shape (N,14)

    return X, y

# ---------------- MODEL ----------------
def cnn_branch(input_shape=(IMG_SIZE, IMG_SIZE, 3), branch_id=0, num_conv=6):
    """Builds one CNN branch with a for-loop for conv layers (num_conv conv layers)."""
    model = models.Sequential(name=f"cnn_branch_{branch_id}")
    for i in range(num_conv):
        filters = 32 * (i // 2 + 1)  # 32,32,64,64,96,96...
        if i == 0:
            model.add(layers.Conv2D(filters, (3,3), activation='relu', padding='same', input_shape=input_shape, name=f"conv_{branch_id}_{i}"))
        else:
            model.add(layers.Conv2D(filters, (3,3), activation='relu', padding='same', name=f"conv_{branch_id}_{i}"))
        if i % 2 == 1:
            model.add(layers.MaxPooling2D((2,2), name=f"pool_{branch_id}_{i}"))
    model.add(layers.GlobalAveragePooling2D(name=f"gap_{branch_id}"))
    return model

def build_multi_input_model(num_conv_per_branch=6):
    inputs, branches = [], []
    for i in range(NUM_INPUTS):
        inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name=f"input_{i+1}")
        # create branch with unique names (cnn_branch uses branch_id in layer names)
        branch = cnn_branch(input_shape=(IMG_SIZE, IMG_SIZE, 3), branch_id=i, num_conv=num_conv_per_branch)(inp)
        inputs.append(inp)
        branches.append(branch)
    merged = layers.Concatenate(name="concat_branches")(branches)
    x = layers.Dense(256, activation='relu', name='dense_merge')(merged)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_LANDMARKS * 2, activation='sigmoid', name='landmarks')(x)
    model = models.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=Huber(), metrics=["mae"])
    return model

# ---------------- TRAINING ----------------
def main():
    ensure_dir(DATA_DIR)
    print("[INFO] Loading dataset from:", DATA_DIR)
    X, y = load_data()
    if X.size == 0:
        print("[ERROR] No training data found in", DATA_DIR)
        return

    N = X.shape[0]
    print(f"[INFO] Loaded {N} samples")

    # Normalize image pixels and prepare inputs as list of arrays: one array per tile index
    # X has shape (N, 9, IMG_SIZE, IMG_SIZE, 3)
    X = X.astype(np.float32) / 255.0
    # Create list inputs: [ (N,H,W,3) for tile 0 ], ...
    inputs = [X[:, i, ...] for i in range(NUM_INPUTS)]

    # Shuffle dataset
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    inputs = [arr[idxs] for arr in inputs]
    y = y[idxs]

    # Train/val split
    split = int(0.8 * N)
    train_inputs = [arr[:split] for arr in inputs]
    val_inputs = [arr[split:] for arr in inputs]
    y_train, y_val = y[:split], y[split:]

    print("[INFO] Train samples:", y_train.shape[0], "Val samples:", y_val.shape[0])

    model = build_multi_input_model(num_conv_per_branch=6)
    model.summary()

    # Fit
    start = time.time()
    model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE
    )
    elapsed = time.time() - start
    print(f"[INFO] Training complete in {elapsed:.1f}s")

    # Save model
    model.save(MODEL_FILE)
    print("[INFO] Saved Keras model to", MODEL_FILE)

    # Attempt TFLite conversion (safe try/except)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(MODEL_TFLITE, "wb") as f:
            f.write(tflite_model)
        print("[INFO] Saved optimized TFLite model to", MODEL_TFLITE)
    except Exception as e:
        print("[WARN] TFLite conversion failed:", e)

if __name__ == "__main__":
    main()
