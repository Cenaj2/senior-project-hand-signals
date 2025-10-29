import cv2, numpy as np, os

GRID = 3
os.makedirs("data", exist_ok=True)

cap = cv2.VideoCapture(0)
ret, prev = cap.read()
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (5,5), 0)

label = input("Gesture label (e.g., left/right/up/down/wave/none): ").strip()
frames = []

print("Recording... press ESC when done.")

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
    frames.append(avg)

    cv2.imshow("diff", diff)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

np.savez(f"data/{label}.npz", X=np.array(frames))
print(f"Saved {len(frames)} samples for '{label}'.")
