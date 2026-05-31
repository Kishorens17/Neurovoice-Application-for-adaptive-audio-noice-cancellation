import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize(noisy_path, restored_path, clean_path=None, sr=16000):
    noisy, _ = librosa.load(noisy_path, sr=sr)
    restored, _ = librosa.load(restored_path, sr=sr)

    min_len = min(len(noisy), len(restored))
    noisy = noisy[:min_len]
    restored = restored[:min_len]

    if clean_path:
        clean, _ = librosa.load(clean_path, sr=sr)
        clean = clean[:min_len]

    # ---------- WAVEFORMS ----------
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.title("Noisy Waveform")
    plt.plot(noisy)

    plt.subplot(3, 1, 2)
    plt.title("Restored Waveform")
    plt.plot(restored)

    if clean_path:
        plt.subplot(3, 1, 3)
        plt.title("Clean Waveform")
        plt.plot(clean)
    else:
        plt.subplot(3, 1, 3)
        plt.title("Difference (Noise Removed)")
        plt.plot(noisy - restored)

    plt.tight_layout()
    plt.show()

    # ---------- SPECTROGRAMS ----------
    noisy_spec = np.abs(librosa.stft(noisy))
    restored_spec = np.abs(librosa.stft(restored))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.title("Noisy Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(noisy_spec, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.title("Restored Spectrogram")
    librosa.display.specshow(librosa.amplitude_to_db(restored_spec, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize(
        noisy_path="noisy.wav",
        restored_path="restored.wav",
        clean_path="data/clean/00000.wav"  # optional
    )
