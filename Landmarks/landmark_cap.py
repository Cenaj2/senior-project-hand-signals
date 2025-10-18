import cv2, json, os, time
import numpy as np

SAVE_DIR = "gesture_data"
GESTURE = input("Enter gesture name: ").strip()
os.makedirs(os.path.join(SAVE_DIR, GESTURE), exist_ok=True)
OUT_JSON = os.path.join(SAVE_DIR, GESTURE, "landmarks.json")

IMG_SIZE = 288
MAX_SAMPLES = 100

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

def main():
    global points
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    samples = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        cv2.imshow("Press [Space] to capture / [Q] to quit", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            points = []
            cv2.imshow("Click 7 landmarks", disp)
            cv2.setMouseCallback("Click 7 landmarks", click_event)
            while len(points) < 7:
                cv2.imshow("Click 7 landmarks", disp)
                if cv2.waitKey(1) & 0xFF == ord("r"):
                    points = []
                    print("[INFO] Reset points.")
            cv2.setMouseCallback("Click 7 landmarks", lambda *args: None)

            img_name = f"{GESTURE}_{idx:04d}.png"
            img_path = os.path.join(SAVE_DIR, GESTURE, img_name)
            cv2.imwrite(img_path, disp)
            norm_pts = np.array(points) / IMG_SIZE
            samples.append({"image": img_name, "landmarks": norm_pts.tolist()})
            print(f"[INFO] Saved {img_name} ({len(samples)} samples).")
            idx += 1

        elif key == ord("q") or idx >= MAX_SAMPLES:
            break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        json.dump({"samples": samples}, open(OUT_JSON, "w"), indent=2)
        print(f"[INFO] Saved {len(samples)} labeled samples → {OUT_JSON}")
    else:
        print("[INFO] No samples captured.")

if __name__ == "__main__":
    main()
