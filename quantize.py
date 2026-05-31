import torch
from model import NanoUNetFiLM, TextEncoder

device = "cpu"  # Quantization runs on CPU

model = NanoUNetFiLM().to(device)
text_encoder = TextEncoder().to(device)

model.load_state_dict(torch.load("nano_unet_film.pth", map_location=device))
text_encoder.load_state_dict(torch.load("text_encoder.pth", map_location=device))

model.eval()
text_encoder.eval()

# Fuse modules if needed (not strictly required here)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), "nano_unet_film_int8.pth")
print("INT8 model saved: nano_unet_film_int8.pth")
