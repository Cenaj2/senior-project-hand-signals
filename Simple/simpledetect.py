import cv2
import numpy as np
import tensorflow as tf
import time
import math

PALM_MODEL = "palm_detection_lite.tflite"
LANDMARK_MODEL = "hand_landmark_lite.tflite"

# --------------------------
# Load palm detector
# --------------------------
palm = tf.lite.Interpreter(model_path=PALM_MODEL)
palm.allocate_tensors()
palm_in = palm.get_input_details()
palm_out = palm.get_output_details()

# --------------------------
# Load hand landmark model
# --------------------------
lm = tf.lite.Interpreter(model_path=LANDMARK_MODEL)
lm.allocate_tensors()
lm_in = lm.get_input_details()
lm_out = lm.get_output_details()

# --------------------------
# Generate anchors (2016 anchors)
# Full BlazePalm grid
# --------------------------
def generate_anchors():
    anchor_dims = [(24, 24), (12, 12), (6, 6), (3,3), (1,1)]
    strides = [8, 16, 32, 64, 128]
    anchors = []

    for (h, w), stride in zip(anchor_dims, strides):
        for y in range(h):
            for x in range(w):
                for _ in range(4):  # 4 anchors per cell
                    cx = (x + 0.5) * stride / 192
                    cy = (y + 0.5) * stride / 192
                    aw = stride / 192
                    ah = stride / 192
                    anchors.append([cx, cy, aw, ah])

    return np.array(anchors, dtype=np.float32)

anchors = generate_anchors()   # 2016 anchors


# --------------------------
# Decode BlazePalm box
# --------------------------
def decode_box(box, anchor):
    cx = box[0] * anchor[2] + anchor[0]
    cy = box[1] * anchor[3] + anchor[1]
    w  = math.exp(box[2]) * anchor[2]
    h  = math.exp(box[3]) * anchor[3]
    return cx, cy, w, h


# --------------------------
# Palm Detection (slow)
# --------------------------
def detect_palm(frame):
    H, W, _ = frame.shape
    ph, pw = palm_in[0]['shape'][1:3]

    img = cv2.resize(frame, (pw, ph))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, ...]

    palm.set_tensor(palm_in[0]["index"], img)
    palm.invoke()

    raw_boxes = palm.get_tensor(palm_out[0]['index'])[0]
    raw_scores = palm.get_tensor(palm_out[1]['index'])[0][:, 0]

    best = np.argmax(raw_scores)
    if raw_scores[best] < 0.25:
        return None

    cx, cy, w, h = decode_box(raw_boxes[best][:4], anchors[best])

    x = int((cx - w/2) * W)
    y = int((cy - h/2) * H)
    bw = int(w * W)
    bh = int(h * H)

    return max(x,0), max(y,0), bw, bh


# --------------------------
# Hand Landmark Tracking (FAST)
# --------------------------
def run_landmarks(roi):
    if roi.size == 0:
        return None

    ih, iw = lm_in[0]["shape"][1:3]
    img = cv2.resize(roi, (iw, ih))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = img[np.newaxis, ...]

    lm.set_tensor(lm_in[0]["index"], img)
    lm.invoke()

    # 21 landmarks × (x,y,z)
    out = lm.get_tensor(lm_out[0]["index"])[0]
    return out


# --------------------------
# Direction Classification
# --------------------------
def classify(px, py, cx, cy, W, H):
    dx = W * 0.15
    dy = H * 0.15

    if py < cy-dy: return "UP"
    if py > cy+dy: return "DOWN"
    if px < cx-dx: return "LEFT"
    if px > cx+dx: return "RIGHT"
    return "CENTER"


# --------------------------
# Main loop
# --------------------------
cap = cv2.VideoCapture(0)
print("Starting… press ESC to quit.")

track_box = None
last_detection_time = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    H, W, _ = frame.shape
    cx, cy = W//2, H//2

    now = time.time()
    direction = "NONE"

    # -----------------------------------
    # 1) Re-detect every 1.5 seconds
    # -----------------------------------
    if track_box is None or (now - last_detection_time) > 1.5:
        det = detect_palm(frame)
        if det:
            track_box = det
            last_detection_time = now
        else:
            track_box = None

    # -----------------------------------
    # 2) Track using fast landmarks
    # -----------------------------------
    if track_box:
        x, y, bw, bh = track_box
        roi = frame[y:y+bh, x:x+bw]

        lm_out = run_landmarks(roi)

        if lm_out is None:
            track_box = None
        else:
            # Wrist landmark index 0
            wx = int(lm_out[0][0] * bw + x)
            wy = int(lm_out[0][1] * bh + y)

            cv2.circle(frame, (wx, wy), 6, (0,0,255), -1)
            direction = classify(wx, wy, cx, cy, W, H)

            # Update tracking box smoothly
            # Expand a little margin to avoid drift
            margin = 80
            nx = max(wx - margin, 0)
            ny = max(wy - margin, 0)
            nbw = min(2*margin, W - nx)
            nbh = min(2*margin, H - ny)
            track_box = (nx, ny, nbw, nbh)

            cv2.rectangle(frame, (nx, ny), (nx+nbw, ny+nbh), (0,255,0), 2)

    # Draw screen center
    cv2.circle(frame, (cx, cy), 8, (255, 0, 0), 2)

    cv2.putText(frame, f"Direction: {direction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Hand Direction Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
