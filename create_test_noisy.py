import librosa
import soundfile as sf
from mixer import mix_audio

clean_path = "data/clean/00107.wav"
noise_path = "data/noise/wind_027.wav"

clean, sr = librosa.load(clean_path, sr=16000)
noise, _ = librosa.load(noise_path, sr=16000)

min_len = min(len(clean), len(noise))
clean = clean[:min_len]
noise = noise[:min_len]

noisy = mix_audio(clean, noise, snr_db=0)

sf.write("noisy.wav", noisy, sr)
print("Created noisy.wav")
