import cv2, numpy as np, joblib

GRID = 3
clf = joblib.load("gesture_model.pkl")

cap = cv2.VideoCapture(0)
ret, prev = cap.read()
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (5,5), 0)

print("Running detection... press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    diff = cv2.absdiff(gray, prev)
    prev = gray

    h, w = diff.shape
    ch, cw = h // GRID, w // GRID
    avg = [np.mean(diff[i*ch:(i+1)*ch, j*cw:(j+1)*cw])
           for i in range(GRID) for j in range(GRID)]

    pred = clf.predict([avg])[0]
    cv2.putText(frame, f"Gesture: {pred}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Hand Motion Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
