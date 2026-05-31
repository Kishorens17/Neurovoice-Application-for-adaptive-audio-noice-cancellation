import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import NanoUNetFiLM, TextEncoder
from preprocess import AudioProcessor

class MobileAudioWrapper(nn.Module):
    def __init__(self, model, text_encoder, n_fft=512, hop_length=128):
        super().__init__()
        self.model = model
        self.text_encoder = text_encoder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, audio, token_id):
        # audio shape: (Batch, Time)
        # STFT
        X = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        
        mag = torch.abs(X)
        phase = torch.angle(X)
        
        # Log mag
        logmag = torch.log1p(mag).unsqueeze(1) # Add channel dim
        
        # Text embedding
        text_emb = self.text_encoder(token_id)
        
        # Model inference
        pred_logmag = self.model(logmag, text_emb)
        pred_logmag = pred_logmag.squeeze(1) # Remove channel dim
        
        # Exp to get magnitude
        pred_mag = torch.expm1(pred_logmag)
        
        # Match phase dim (center crop)
        Wa = pred_mag.shape[-1]
        Wb = phase.shape[-1]
        start_w = (Wb - Wa) // 2
        phase_cropped = phase[..., start_w:start_w + Wa]
        
        # ISTFT
        complex_spec = pred_mag * torch.exp(1j * phase_cropped)
        restored = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window
        )
        
        return restored

def export_mobile_wrapper():
    model = NanoUNetFiLM()
    text_encoder = TextEncoder()
    
    # Load weights
    try:
        model.load_state_dict(torch.load("../nano_unet_film.pth", map_location="cpu"))
        text_encoder.load_state_dict(torch.load("../text_encoder.pth", map_location="cpu"))
        print("Weights loaded.")
    except Exception as e:
        print(f"Weights not loaded: {e}")

    wrapper = MobileAudioWrapper(model, text_encoder)
    wrapper.eval()

    # Dummy input: 2 seconds @ 16kHz = 32000 samples
    dummy_audio = torch.randn(1, 32000)
    dummy_token = torch.tensor([0])

    # Script/Trace
    traced_wrapper = torch.jit.trace(wrapper, (dummy_audio, dummy_token))
    
    # Save as standard .pt for better compatibility
    traced_wrapper.save("audio_enhancer_mobile.pt")
    print("Exported audio_enhancer_mobile.pt")
    
    # Copy to android assets
    import shutil
    target_path = "../android_app/app/src/main/assets/audio_enhancer_mobile.pt"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy("audio_enhancer_mobile.pt", target_path)
    print(f"Copied to {target_path}")

if __name__ == "__main__":
    export_mobile_wrapper()
