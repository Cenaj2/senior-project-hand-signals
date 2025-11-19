#!/usr/bin/env python3
# collect_data.py
# Record per-frame GRID-average motion features and save to data/<label>_<timestamp>.npz

import cv2
import numpy as np
import os
import time
import sys

GRID = 3
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Unable to open camera (device 0).")
    sys.exit(1)

# read first frame safely
ret, prev = cap.read()
if not ret or prev is None:
    print("ERROR: Unable to read first frame from camera.")
    cap.release()
    sys.exit(1)

prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (5, 5), 0)

label = input("Gesture label (e.g., left/right/up/down/wave/none): ").strip()
if not label:
    print("No label provided — exiting.")
    cap.release()
    sys.exit(1)

frames = []
print("Recording... press ESC to stop, or Ctrl-C to abort.")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("WARNING: frame read failed, stopping.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, prev)
        prev = gray

        h, w = diff.shape
        # ensure GRID splits are at least 1 pixel
        if h < GRID or w < GRID:
            print(f"ERROR: Frame too small for GRID={GRID} (frame {w}x{h}).")
            break

        ch, cw = h // GRID, w // GRID
        # In rare edge cases ch or cw might be 0; guard it
        if ch == 0 or cw == 0:
            print(f"ERROR: Computed chunk size is zero: ch={ch}, cw={cw}.")
            break

        avg = []
        for i in range(GRID):
            for j in range(GRID):
                y0, y1 = i * ch, (i + 1) * ch if i < GRID - 1 else h
                x0, x1 = j * cw, (j + 1) * cw if j < GRID - 1 else w
                region = diff[y0:y1, x0:x1]
                avg.append(float(np.mean(region)))

        frames.append(avg)

        cv2.imshow("diff", diff)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
except KeyboardInterrupt:
    print("\nInterrupted by user, saving what we have...")

cap.release()
cv2.destroyAllWindows()

if len(frames) == 0:
    print("No frames recorded; nothing saved.")
else:
    ts = int(time.time())
    out_path = os.path.join(OUT_DIR, f"{label}_{ts}.npz")
    X = np.array(frames, dtype=np.float32)
    np.savez_compressed(out_path, X=X, label=label, frames=X.shape[0])
    print(f"Saved {X.shape[0]} frames to '{out_path}'.")
