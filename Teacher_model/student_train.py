import os, json, cv2, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "dataset"
VIDEO_PATH = os.path.join(DATA_DIR, "hand_video.mp4")
JSON_PATH  = os.path.join(DATA_DIR, "landmarks.json")
IMG_SIZE = 160       # bigger input improves accuracy
BATCH_SIZE = 16
EPOCHS = 50

# -------------------------------
# Load labels
# -------------------------------
with open(JSON_PATH) as f: labels = json.load(f)
frame_dict = {d["frame"]: d["coords"] for d in labels}

# -------------------------------
# Load frames + normalize coords
# -------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frames, coords = [], []
fid = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    if fid in frame_dict:
        h, w, _ = frame.shape
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)/255.0
        frames.append(frame)

        # normalize coordinates to 0–1 range
        arr = np.array(frame_dict[fid]) / [w, h, 1]
        coords.append(arr.flatten())
    fid += 1
cap.release()

X, y = np.stack(frames), np.stack(coords)
print(f"✅ {len(X)} labeled frames loaded")

# -------------------------------
# Split train/test
# -------------------------------
split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# Define transfer-learned model
# -------------------------------
base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                         include_top=False, weights='imagenet')
base.trainable = False  # freeze backbone

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(21)  # 7 points × (x,y,z)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# -------------------------------
# Train
# -------------------------------
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

# -------------------------------
# Save models
# -------------------------------
model.save(os.path.join(DATA_DIR, "student_model.h5"))
print("Saved student_model.h5")

# Quantized TFLite export for Pi
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
with open(os.path.join(DATA_DIR, "student_model.tflite"), "wb") as f:
    f.write(tflite)
print("Saved quantized student_model.tflite")
