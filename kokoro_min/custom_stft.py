import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomSTFT(nn.Module):
    """
    Conv-based STFT / iSTFT (no complex tensors)

    - forward: conv1d (real + imag)
    - inverse: conv_transpose1d
    - deterministic
    - ONNX / TorchScript friendly
    """

    def __init__(
        self,
        filter_length=800,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="replicate",
    ):
        super().__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode

        self.freq_bins = self.n_fft // 2 + 1

        # -------------------------
        # Window
        # -------------------------
        if window != "hann":
            raise ValueError("Only Hann window is supported")

        win = torch.hann_window(
            win_length,
            periodic=True,
            dtype=torch.float32,
        )

        if win_length < self.n_fft:
            win = F.pad(win, (0, self.n_fft - win_length))
        elif win_length > self.n_fft:
            win = win[: self.n_fft]

        self.register_buffer("window", win)

        # -------------------------
        # Forward DFT kernels
        # -------------------------
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)

        angle = 2 * np.pi * np.outer(k, n) / self.n_fft
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)

        fw = win.numpy()
        real_kernel = torch.from_numpy(dft_real * fw).float().unsqueeze(1)
        imag_kernel = torch.from_numpy(dft_imag * fw).float().unsqueeze(1)

        self.register_buffer("weight_forward_real", real_kernel)
        self.register_buffer("weight_forward_imag", imag_kernel)

        # -------------------------
        # Inverse DFT kernels
        # -------------------------
        inv_scale = 1.0 / self.n_fft
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft

        idft_cos = np.cos(angle_t).T
        idft_sin = np.sin(angle_t).T

        bw = win.numpy() * inv_scale
        real_inv = torch.from_numpy(idft_cos * bw).float().unsqueeze(1)
        imag_inv = torch.from_numpy(idft_sin * bw).float().unsqueeze(1)

        self.register_buffer("weight_backward_real", real_inv)
        self.register_buffer("weight_backward_imag", imag_inv)

    # =================================================
    # STFT
    # =================================================
    def transform(self, waveform: torch.Tensor):
        """
        waveform: (B, T)
        returns: magnitude, phase
        """
        if self.center:
            pad = self.n_fft // 2
            waveform = F.pad(waveform, (pad, pad), mode=self.pad_mode)

        x = waveform.unsqueeze(1)

        real = F.conv1d(
            x,
            self.weight_forward_real,
            stride=self.hop_length,
        )
        imag = F.conv1d(
            x,
            self.weight_forward_imag,
            stride=self.hop_length,
        )

        mag = torch.sqrt(real**2 + imag**2 + 1e-9)
        phase = torch.atan2(imag, real)

        return mag, phase

    # =================================================
    # iSTFT
    # =================================================
    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        real_rec = F.conv_transpose1d(
            real,
            self.weight_backward_real,
            stride=self.hop_length,
        )
        imag_rec = F.conv_transpose1d(
            imag,
            self.weight_backward_imag,
            stride=self.hop_length,
        )

        waveform = real_rec - imag_rec

        if self.center:
            pad = self.n_fft // 2
            waveform = waveform[..., pad:-pad]

        if length is not None:
            waveform = waveform[..., :length]

        return waveform.squeeze(1)

    # =================================================
    def forward(self, x: torch.Tensor):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])
