import os
import random
import librosa
import torch
from torch.utils.data import Dataset
from mixer import mix_audio
from prompt_utils import NOISE_CLASSES

FIXED_LENGTH = 32000  # 2 seconds at 16kHz

def fix_length(x, length):
    if len(x) > length:
        return x[:length]
    elif len(x) < length:
        return torch.nn.functional.pad(x, (0, length - len(x)))
    return x

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sr=16000):
        self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)]
        self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)]
        self.sr = sr

    def __len__(self):
        return min(len(self.clean_files), len(self.noise_files))

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        noise_path = random.choice(self.noise_files)

        clean, _ = librosa.load(clean_path, sr=self.sr)
        noise, _ = librosa.load(noise_path, sr=self.sr)

        min_len = min(len(clean), len(noise))
        clean = clean[:min_len]
        noise = noise[:min_len]

        snr = random.uniform(-5, 10)
        mixed = mix_audio(clean, noise, snr)

        # Infer class from filename
        label = 0
        for i, cls in enumerate(NOISE_CLASSES):
            if cls in noise_path.lower():
                label = i
                break

        mixed = torch.tensor(mixed, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)

        mixed = fix_length(mixed, FIXED_LENGTH)
        clean = fix_length(clean, FIXED_LENGTH)

        return mixed, clean, torch.tensor(label, dtype=torch.long)

