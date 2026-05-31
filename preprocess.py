import torch

class AudioProcessor:
    def __init__(self, n_fft=512, hop_length=128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

    def stft(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True
        )

    def istft(self, X):
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(X.device)
        )

    def mag_phase(self, X):
        return torch.abs(X), torch.angle(X)

    def log_mag(self, mag):
        return torch.log1p(mag)
