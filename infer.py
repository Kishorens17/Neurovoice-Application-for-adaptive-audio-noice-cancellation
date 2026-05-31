import torch
import librosa
import soundfile as sf
from preprocess import AudioProcessor
from model import NanoUNetFiLM, TextEncoder
from prompt_utils import prompt_to_token

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AudioProcessor()

model = NanoUNetFiLM().to(device)
text_encoder = TextEncoder().to(device)

model.load_state_dict(torch.load("nano_unet_film.pth", map_location=device))
text_encoder.load_state_dict(torch.load("text_encoder.pth", map_location=device))

model.eval()
text_encoder.eval()

def match_time_dim(a, b):
    Wa = a.shape[-1]
    Wb = b.shape[-1]
    start_w = (Wb - Wa) // 2
    return b[..., :, start_w:start_w + Wa]

def restore_audio(noisy_path, prompt, out_path="restored.wav"):
    audio, sr = librosa.load(noisy_path, sr=16000)
    x = torch.tensor(audio).to(device)

    X = processor.stft(x)
    mag, phase = processor.mag_phase(X)
    logmag = processor.log_mag(mag)

    inp = logmag.unsqueeze(0).unsqueeze(0)

    token_id = prompt_to_token(prompt)
    token = torch.tensor([token_id]).to(device)
    text_emb = text_encoder(token)

    with torch.no_grad():
        pred_mag = model(inp, text_emb)

    pred_mag = pred_mag.squeeze(0).squeeze(0)

    phase = match_time_dim(pred_mag, phase)
    complex_spec = pred_mag * torch.exp(1j * phase)

    restored = processor.istft(complex_spec)
    restored = restored.cpu().numpy()

    restored = restored / (abs(restored).max() + 1e-8)
    sf.write(out_path, restored, sr)

    print("Saved:", out_path)

# ============ RUN ============
if __name__ == "__main__":
    noisy_file = "noisy.wav"
    prompt = input("Enter prompt: ")
    restore_audio(noisy_file, prompt)
