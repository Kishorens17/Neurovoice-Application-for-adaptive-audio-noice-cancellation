import librosa
import numpy as np
from pesq import pesq
from pystoi import stoi
from mir_eval.separation import bss_eval_sources

def evaluate_metrics(clean_path, restored_path, sr=16000):
    clean, _ = librosa.load(clean_path, sr=sr)
    restored, _ = librosa.load(restored_path, sr=sr)

    min_len = min(len(clean), len(restored))
    clean = clean[:min_len]
    restored = restored[:min_len]

    pesq_score = pesq(sr, clean, restored, 'wb')
    stoi_score = stoi(clean, restored, sr, extended=False)

    sdr, sir, sar, _ = bss_eval_sources(
        clean.reshape(1, -1),
        restored.reshape(1, -1)
    )

    print("PESQ:", pesq_score)
    print("STOI:", stoi_score)
    print("SDR:", sdr[0])

if __name__ == "__main__":
    clean_file = "data/clean/00000.wav"
    restored_file = "restored.wav"
    evaluate_metrics(clean_file, restored_file)
