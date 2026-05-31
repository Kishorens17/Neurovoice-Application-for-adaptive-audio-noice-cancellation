import os
import pandas as pd
import shutil
from tqdm import tqdm

# CHANGE THIS IF NEEDED
ESC_ROOT = r"C:\Users\kisho\Downloads\ESC-50-master\ESC-50-master"
AUDIO_DIR = os.path.join(ESC_ROOT, "audio")
META_FILE = os.path.join(ESC_ROOT, "meta", "esc50.csv")

OUTPUT_DIR = "data/noise"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classes we want
TARGET_CLASSES = [
    "dog",
    "siren",
    "rain",
    "wind",
    "engine",
    "keyboard",
    "footsteps",
    "door",
    "baby",
    "clock"
]

df = pd.read_csv(META_FILE)

counters = {cls: 0 for cls in TARGET_CLASSES}

print("Preparing ESC-50 noise files...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    category = row["category"].lower()
    filename = row["filename"]

    for cls in TARGET_CLASSES:
        if cls in category:
            src_path = os.path.join(AUDIO_DIR, filename)

            if not os.path.exists(src_path):
                continue

            out_name = f"{cls}_{counters[cls]:03d}.wav"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            shutil.copy(src_path, out_path)
            counters[cls] += 1

print("Done!")
print("Counts:")
for cls, count in counters.items():
    print(cls, ":", count)
