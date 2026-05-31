import pytest
import torch
from model import NanoUNetFiLM, TextEncoder
from preprocess import AudioProcessor

@pytest.fixture
def processor():
    return AudioProcessor(n_fft=512, hop_length=128)

@pytest.fixture
def model():
    return NanoUNetFiLM(text_dim=64)

@pytest.fixture
def text_encoder():
    return TextEncoder(vocab_size=100, emb_dim=64)

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
