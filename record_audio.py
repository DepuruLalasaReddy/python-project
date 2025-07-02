# record_audio.py
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename, duration=5, fs=44100):
    print("🎙️ Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print(f"✅ Recording saved: {filename}")

# Run to record truth and lie samples
if __name__ == "__main__":
    for i in range(1, 4):
        record_audio(f"dataset/truth{i}.wav")
    for i in range(1, 4):
        record_audio(f"dataset/lie{i}.wav")
