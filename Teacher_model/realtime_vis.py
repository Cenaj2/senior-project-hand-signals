import cv2, numpy as np, tensorflow as tf, collections

MODEL_PATH = "student_model.tflite"
IMG_SIZE = 160

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
history = collections.deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret: break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img/255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    landmarks = np.clip(preds.reshape(-1, 3), 0, 1)

    # temporal smoothing
    history.append(landmarks)
    landmarks = np.mean(history, axis=0)

    # draw
    h, w, _ = frame.shape
    for (x, y, z) in landmarks:
        cx, cy = int(x * w), int(y * h)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    cv2.imshow("Pi Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
