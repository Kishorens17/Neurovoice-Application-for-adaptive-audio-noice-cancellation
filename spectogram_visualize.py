import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def show_spectrograms(noisy_path, restored_path, clean_path=None, sr=16000):
    noisy, _ = librosa.load(noisy_path, sr=sr)
    restored, _ = librosa.load(restored_path, sr=sr)

    min_len = min(len(noisy), len(restored))
    noisy = noisy[:min_len]
    restored = restored[:min_len]

    # STFT
    noisy_spec = np.abs(librosa.stft(noisy))
    restored_spec = np.abs(librosa.stft(restored))

    noisy_db = librosa.amplitude_to_db(noisy_spec, ref=np.max)
    restored_db = librosa.amplitude_to_db(restored_spec, ref=np.max)

    plt.figure(figsize=(14, 10))

    plt.subplot(3 if clean_path else 2, 1, 1)
    plt.title("Noisy Spectrogram")
    librosa.display.specshow(noisy_db, sr=sr, y_axis='log', x_axis='time')
    plt.colorbar()

    plt.subplot(3 if clean_path else 2, 1, 2)
    plt.title("Restored Spectrogram")
    librosa.display.specshow(restored_db, sr=sr, y_axis='log', x_axis='time')
    plt.colorbar()

    if clean_path:
        clean, _ = librosa.load(clean_path, sr=sr)
        clean = clean[:min_len]
        clean_spec = np.abs(librosa.stft(clean))
        clean_db = librosa.amplitude_to_db(clean_spec, ref=np.max)

        plt.subplot(3, 1, 3)
        plt.title("Clean Spectrogram")
        librosa.display.specshow(clean_db, sr=sr, y_axis='log', x_axis='time')
        plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_spectrograms(
        noisy_path="noisy.wav",
        restored_path="restored.wav",
        clean_path="data/clean/00000.wav"  # optional
    )
