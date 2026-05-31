import torch
import numpy as np
from preprocess import AudioProcessor

def test_stft_istft_reconstruction(processor):
    # Create a simple signal
    sr = 16000
    t = torch.linspace(0, 1, sr)
    x = torch.sin(2 * torch.pi * 440 * t)
    
    # Forward
    X = processor.stft(x)
    assert X.is_complex()
    
    # Backward
    x_hat = processor.istft(X)
    
    # Match lengths
    min_len = min(len(x), len(x_hat))
    x = x[:min_len]
    x_hat = x_hat[:min_len]
    
    # Check reconstruction error (L1)
    error = torch.mean(torch.abs(x - x_hat))
    assert error < 1e-5

def test_mag_phase(processor):
    x = torch.randn(16000)
    X = processor.stft(x)
    mag, phase = processor.mag_phase(X)
    
    assert torch.all(mag >= 0)
    assert mag.shape == X.shape
    assert phase.shape == X.shape

def test_log_mag(processor):
    mag = torch.tensor([0.0, 1.0, 2.0])
    logmag = processor.log_mag(mag)
    expected = torch.log1p(mag)
    assert torch.allclose(logmag, expected)
