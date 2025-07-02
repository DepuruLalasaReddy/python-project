# extract_features.py
import librosa
import numpy as np

def extract_features(file_name):
    audio, sr = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs
