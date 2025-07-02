# train_model.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from extract_features import extract_features

features = []
labels = []

for file in os.listdir("dataset"):
    if file.endswith(".wav"):
        label = 1 if "lie" in file.lower() else 0
        data = extract_features(f"dataset/{file}")
        features.append(data)
        labels.append(label)

df = pd.DataFrame(features)
df['label'] = labels

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Model Trained. Accuracy: {acc*100:.2f}%")

joblib.dump(model, "lie_detector_model.pkl")
