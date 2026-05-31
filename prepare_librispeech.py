import os
import librosa
import soundfile as sf
from tqdm import tqdm

# CHANGE THIS IF NEEDED
LIBRI_ROOT = r"C:\Users\kisho\Downloads\train-clean-100\LibriSpeech\train-clean-100"
OUTPUT_DIR = "data/clean"
TARGET_SR = 16000
MAX_FILES = 500  # limit for now (for fast testing)

os.makedirs(OUTPUT_DIR, exist_ok=True)

counter = 0

print("Scanning LibriSpeech...")

for root, dirs, files in os.walk(LIBRI_ROOT):
    for file in files:
        if file.endswith(".flac"):
            flac_path = os.path.join(root, file)

            try:
                audio, sr = librosa.load(flac_path, sr=TARGET_SR)
            except Exception as e:
                print(f"Skipping {flac_path}: {e}")
                continue

            out_name = f"{counter:05d}.wav"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            sf.write(out_path, audio, TARGET_SR)

            counter += 1

            if counter >= MAX_FILES:
                break

    if counter >= MAX_FILES:
        break

print(f"Done! {counter} files written to {OUTPUT_DIR}")
