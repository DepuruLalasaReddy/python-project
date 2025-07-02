# test_lie_detector.py
from record_audio import record_audio
from extract_features import extract_features
import joblib

model = joblib.load("lie_detector_model.pkl")

# Record new sample to test
record_audio("test_sample.wav", duration=5)

# Predict
features = extract_features("test_sample.wav")
result = model.predict([features])

if result[0] == 1:
    print("🟥 Prediction: LIE detected!")
else:
    print("🟩 Prediction: Truth detected!")
