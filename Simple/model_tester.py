import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "palm_detection_lite.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== MODEL INPUT INFO ===")
print(input_details[0])

print("\n=== MODEL OUTPUT INFO ===")
for i, od in enumerate(output_details):
    print(f"Output[{i}]: {od}")

# Open camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Camera failed")

h, w, _ = frame.shape
input_h = input_details[0]['shape'][1]
input_w = input_details[0]['shape'][2]

# Preprocess: just resize (NO crop)
img = cv2.resize(frame, (input_w, input_h))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, 0)

interpreter.set_tensor(input_details[0]["index"], img)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]["index"])[0]  # [294, 18]
scores = interpreter.get_tensor(output_details[1]["index"])[0][:, 0]

print("\n=== RAW BOX SHAPE ===")
print(boxes.shape)

print("\n=== RAW SCORE SHAPE ===")
print(scores.shape)

# Print first anchor box
print("\n=== FIRST ANCHOR RAW 18 VALUES ===")
print(boxes[0])

# Find best-scoring anchor
best = int(np.argmax(scores))
print("\n=== TOP SCORING ANCHOR INDEX ===")
print(best)

print("\n=== TOP ANCHOR SCORE ===")
print(scores[best])

print("\n=== TOP ANCHOR 18 RAW VALUES ===")
print(boxes[best])
