import numpy as np, glob, joblib
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = [], []
for f in glob.glob("data/*.npz"):
    data = np.load(f)
    label = os.path.splitext(os.path.basename(f))[0]
    X.extend(data["X"])
    y.extend([label]*len(data["X"]))

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LinearSVC()
clf.fit(X_train, y_train)
print("Accuracy Report:")
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, "gesture_model.pkl")
print("Model saved as gesture_model.pkl")
