import torch
import torch.nn as nn

# =========================
# Center crop for skip connections
# Crops only TIME dimension (width), NOT frequency
# =========================
def center_crop(enc_feat, target_feat):
    _, _, Ht, Wt = target_feat.shape
    _, _, He, We = enc_feat.shape

    # Frequency dimension must match (do not crop)
    # Crop ONLY time dimension
    start_w = (We - Wt) // 2

    return enc_feat[:, :, :, start_w:start_w + Wt]

# =========================
# FiLM Layer
# =========================
class FiLM(nn.Module):
    def __init__(self, text_dim, channels):
        super().__init__()
        self.gamma = nn.Linear(text_dim, channels)
        self.beta = nn.Linear(text_dim, channels)

    def forward(self, x, t):
        gamma = self.gamma(t).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(t).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

# =========================
# Text Encoder
# =========================
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=100, emb_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, tokens):
        return self.embedding(tokens)

# =========================
# Depthwise Separable Conv
# =========================
class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.pointwise(self.depthwise(x)))

# =========================
# Nano U-Net + FiLM (SAFE VERSION)
# =========================
class NanoUNetFiLM(nn.Module):
    def __init__(self, text_dim=64):
        super().__init__()

        # Encoder
        self.enc1 = SepConv(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # time only

        self.enc2 = SepConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.enc3 = SepConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))

        # Bottleneck
        self.bottleneck = SepConv(128, 256)
        self.film = FiLM(text_dim, 256)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=(1, 2), mode="nearest")
        self.dec1 = SepConv(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=(1, 2), mode="nearest")
        self.dec2 = SepConv(128 + 64, 64)

        self.up3 = nn.Upsample(scale_factor=(1, 2), mode="nearest")
        self.dec3 = SepConv(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck + FiLM
        b = self.bottleneck(p3)
        b = self.film(b, t)

        # Decoder
        d1 = self.up1(b)
        e3c = center_crop(e3, d1)
        d1 = self.dec1(torch.cat([d1, e3c], dim=1))

        d2 = self.up2(d1)
        e2c = center_crop(e2, d2)
        d2 = self.dec2(torch.cat([d2, e2c], dim=1))

        d3 = self.up3(d2)
        e1c = center_crop(e1, d3)
        d3 = self.dec3(torch.cat([d3, e1c], dim=1))

        return self.out(d3)
