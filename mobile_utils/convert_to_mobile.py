import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import NanoUNetFiLM

def convert_to_torchscript():
    model = NanoUNetFiLM()
    
    # Load weights if they exist
    try:
        model.load_state_dict(torch.load("nano_unet_film.pth", map_location="cpu"))
        print("Loaded weights from nano_unet_film.pth")
    except Exception as e:
        print(f"Weights not loaded: {e}")

    model.eval()

    # Define dummy inputs for tracing
    # Shape: (Batch, Channel, Freq, Time)
    dummy_mag = torch.randn(1, 1, 257, 251) 
    dummy_text = torch.randn(1, 64)

    # Use tracing for conversion
    traced_model = torch.jit.trace(model, (dummy_mag, dummy_text))
    
    # Save optimized model
    traced_model.save("nano_unet_film_mobile.ptl")
    print("Saved TorchScript model to nano_unet_film_mobile.ptl")

if __name__ == "__main__":
    convert_to_torchscript()
