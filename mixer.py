import numpy as np
import random

def rms(x):
    return np.sqrt(np.mean(x**2))

def mix_audio(clean, noise, snr_db):
    Ps = rms(clean)**2
    Pn = rms(noise)**2
    alpha = np.sqrt(Ps / (Pn * 10**(snr_db / 10)))
    noise_scaled = alpha * noise
    mixed = clean + noise_scaled
    return mixed
