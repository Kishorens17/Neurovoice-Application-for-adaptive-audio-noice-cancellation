import os
import torch
import numpy as np
import pytest
from dataset import AudioDataset, fix_length
from mixer import mix_audio

def test_fix_length():
    x = torch.randn(100)
    # Pad
    padded = fix_length(x, 150)
    assert len(padded) == 150
    assert torch.all(padded[100:] == 0)
    
    # Crop
    cropped = fix_length(x, 50)
    assert len(cropped) == 50

def test_mix_audio():
    clean = np.random.randn(16000)
    noise = np.random.randn(16000)
    snr = 10
    mixed = mix_audio(clean, noise, snr)
    assert mixed.shape == clean.shape
    assert not np.array_equal(mixed, clean)

def test_audio_dataset():
    # Requires data/clean and data/noise to exist from generate_dummy_data.py
    clean_dir = "data/clean"
    noise_dir = "data/noise"
    
    if not os.path.exists(clean_dir) or not os.path.exists(noise_dir):
        pytest.skip("Dummy data not found")
        
    dataset = AudioDataset(clean_dir, noise_dir)
    assert len(dataset) > 0
    
    mixed, clean, label = dataset[0]
    assert isinstance(mixed, torch.Tensor)
    assert isinstance(clean, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert mixed.shape == (32000,)
    assert clean.shape == (32000,)
