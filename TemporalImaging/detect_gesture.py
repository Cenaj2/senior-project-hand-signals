#!/usr/bin/env python3
# detect_gesture.py
# Loads gesture_pipeline.pkl and runs live detection from camera.
# Uses a short rolling majority vote to stabilize predictions.

import cv2
import numpy as np
import joblib
from collections import deque
import sys
import time
import os

GRID = 3
MODEL_PATH = "gesture_pipeline.pkl"
SMOOTH_N = 5  # number of recent predictions to majority-vote

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: model not found at '{MODEL_PATH}'. Run train_model.py first.")
    sys.exit(1)

pipeline = joblib.load(MODEL_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Unable to open camera (device 0).")
    sys.exit(1)

ret, prev = cap.read()
if not ret or prev is None:
    print("ERROR: Unable to read first frame from camera.")
    cap.release()
    sys.exit(1)

prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (5, 5), 0)

preds = deque(maxlen=SMOOTH_N)
print("Running detection... press ESC to exit.")

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
        if h < GRID or w < GRID:
            cv2.putText(frame, f"Frame too small for GRID={GRID}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Hand Motion Detector", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        ch, cw = h // GRID, w // GRID
        if ch == 0 or cw == 0:
            cv2.putText(frame, "Chunk size 0 (increase resolution)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Hand Motion Detector", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        avg = []
        for i in range(GRID):
            for j in range(GRID):
                y0, y1 = i * ch, (i + 1) * ch if i < GRID - 1 else h
                x0, x1 = j * cw, (j + 1) * cw if j < GRID - 1 else w
                region = diff[y0:y1, x0:x1]
                avg.append(float(np.mean(region)))

        # predict
        try:
            pred = pipeline.predict([np.array(avg, dtype=np.float32)])[0]
        except Exception as e:
            pred = f"ERR:{e}"

        preds.append(pred)
        # majority vote
        if len(preds) > 0:
            # pick the most common
            majority = max(set(preds), key=lambda x: preds.count(x))
        else:
            majority = pred

        cv2.putText(frame, f"Gesture: {majority}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Hand Motion Detector", frame)

        if cv2.waitKey(1) == 27:
            break
except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
