import os
import numpy as np
import soundfile as sf

os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/noise", exist_ok=True)

sr = 16000
duration = 2  # seconds
t = np.linspace(0, duration, int(sr * duration))

# Generate clean signals (sine waves)
for i in range(10):
    freq = np.random.randint(200, 600)
    clean = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(f"data/clean/clean_{i}.wav", clean, sr)

# Generate noise signals
noise_types = ["dog", "siren", "rain", "wind"]

for ntype in noise_types:
    for i in range(5):
        noise = np.random.randn(len(t)) * 0.3
        sf.write(f"data/noise/{ntype}_{i}.wav", noise, sr)

print("Dummy dataset created!")
