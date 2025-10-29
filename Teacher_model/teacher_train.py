import cv2, os, json
import mediapipe as mp

SAVE_DIR = "dataset"
VIDEO_PATH = os.path.join(SAVE_DIR, "hand_video.mp4")
JSON_PATH  = os.path.join(SAVE_DIR, "landmarks.json")
FPS = 30

os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS,
                      (int(cap.get(3)), int(cap.get(4))))

data = []
frame_id = 0
print("🎥 Recording... Press ESC to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # palm + 5 fingertips + wrist center (7 total)
            selected = [hand_landmarks.landmark[i] for i in [0, 4, 8, 12, 16, 20, 9]]
            coords = [[lm.x * w, lm.y * h, lm.z] for lm in selected]
            data.append({"frame": frame_id, "coords": coords})
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    out.write(frame)
    cv2.imshow("Teacher Labeling", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
hands.close()

with open(JSON_PATH, "w") as f: json.dump(data, f, indent=2)
print(f"✅ Saved {len(data)} labeled frames to {JSON_PATH}")
