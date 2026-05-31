import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AudioDataset
from preprocess import AudioProcessor
from model import NanoUNetFiLM, TextEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= CONFIG =================
CLEAN_DIR = "data/clean"
NOISE_DIR = "data/noise"
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-3

# =========================================

processor = AudioProcessor()
dataset = AudioDataset(CLEAN_DIR, NOISE_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = NanoUNetFiLM().to(device)
text_encoder = TextEncoder().to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(text_encoder.parameters()), lr=LR
)
loss_fn = nn.L1Loss()

def audio_to_logmag(x):
    X = processor.stft(x)
    mag, _ = processor.mag_phase(X)
    return processor.log_mag(mag)

def match_time_dim(a, b):
    Wa = a.shape[-1]
    Wb = b.shape[-1]
    start_w = (Wb - Wa) // 2
    return b[..., :, start_w:start_w + Wa]

print("Training started...")

for epoch in range(EPOCHS):
    total_loss = 0

    for mixed, clean, labels in loader:
        mixed = mixed.to(device)
        clean = clean.to(device)
        labels = labels.to(device)

        mixed_mag = audio_to_logmag(mixed).unsqueeze(1)
        clean_mag = audio_to_logmag(clean).unsqueeze(1)

        text_emb = text_encoder(labels)

        pred = model(mixed_mag, text_emb)

        clean_mag = match_time_dim(pred, clean_mag)
        loss = loss_fn(pred, clean_mag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "nano_unet_film.pth")
torch.save(text_encoder.state_dict(), "text_encoder.pth")

print("Model saved.")
