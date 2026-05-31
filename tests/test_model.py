import torch
import pytest
from model import SepConv, FiLM, NanoUNetFiLM

def test_sep_conv_shape():
    conv = SepConv(in_ch=1, out_ch=32)
    x = torch.randn(1, 1, 64, 128)
    out = conv(x)
    assert out.shape == (1, 32, 64, 128)

def test_film_layer():
    film = FiLM(text_dim=64, channels=128)
    x = torch.randn(1, 128, 16, 32)
    t = torch.randn(1, 64)
    out = film(x, t)
    assert out.shape == x.shape

def test_nano_unet_output_shape(model):
    x = torch.randn(1, 1, 257, 251) # STFT shape for 32000 samples @ 512 n_fft
    t = torch.randn(1, 64)
    out = model(x, t)
    
    # U-Net output shape should match input freq dim usually,
    # but time dim might be cropped if not power of 2
    assert out.shape[1] == 1
    assert out.shape[2] == x.shape[2]
    assert out.shape[3] <= x.shape[3]
