#!/usr/bin/env python3
# train_model.py
# Loads all data/*.npz, trains a pipeline (StandardScaler -> LinearSVC), and saves pipeline.

import numpy as np
import glob
import joblib
import os
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_GLOB = "data/*.npz"
OUT_MODEL = "gesture_pipeline.pkl"

X_all = []
y_all = []

for f in glob.glob(DATA_GLOB):
    try:
        d = np.load(f, allow_pickle=True)
        X = d["X"]
        # label stored either as d['label'] or filename base
        if "label" in d.files:
            label = str(d["label"].tolist())
        else:
            label = os.path.splitext(os.path.basename(f))[0]
        X_all.extend(X)
        y_all.extend([label] * len(X))
        print(f"Loaded {len(X)} frames for label '{label}' from {f}")
    except Exception as e:
        print(f"WARNING: couldn't load {f}: {e}")

if len(X_all) == 0:
    print("ERROR: No training data found. Run collect_data.py to create data/*.npz files.")
    raise SystemExit(1)

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearSVC(max_iter=5000, random_state=42))
])

print("Training classifier...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, OUT_MODEL)
print(f"Saved model pipeline to {OUT_MODEL}")
