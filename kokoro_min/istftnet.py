import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ LOCAL import (FIX)
from .custom_stft import CustomSTFT


# =====================================================
# AdaIN Residual Block (1D)
# =====================================================
class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        style_dim,
        upsample=False,
        dropout_p=0.0,
    ):
        super().__init__()
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out

        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, padding=1)

        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

        self.dropout = nn.Dropout(dropout_p)

        if self.learned_sc:
            self.conv_sc = nn.Conv1d(dim_in, dim_out, 1)

    def _residual(self, x, s):
        h = self.norm1(x, s)
        h = F.leaky_relu(h, 0.2)

        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        h = self.conv1(h)
        h = self.norm2(h, s)
        h = F.leaky_relu(h, 0.2)
        h = self.dropout(h)
        h = self.conv2(h)
        return h

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv_sc(x)
        return x

    def forward(self, x, s):
        return self._shortcut(x) + self._residual(x, s)


# =====================================================
# Adaptive Instance Norm
# =====================================================
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s).unsqueeze(-1)
        gamma, beta = torch.chunk(h, 2, dim=1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-5
        return (x - mean) / std * gamma + beta


# =====================================================
# Decoder (iSTFTNet)
# =====================================================
class Decoder(nn.Module):
    def __init__(
        self,
        dim_in,
        style_dim,
        dim_out,
        disable_complex=False,
        **istftnet_cfg,
    ):
        super().__init__()

        self.disable_complex = disable_complex

        self.stft = CustomSTFT(**istftnet_cfg["stft"])

        self.pre = nn.Conv1d(dim_in, istftnet_cfg["hidden_dim"], 1)

        self.blocks = nn.ModuleList()
        for _ in range(istftnet_cfg["n_resblock"]):
            self.blocks.append(
                AdainResBlk1d(
                    istftnet_cfg["hidden_dim"],
                    istftnet_cfg["hidden_dim"],
                    style_dim,
                    upsample=True,
                    dropout_p=istftnet_cfg.get("dropout", 0.0),
                )
            )

        self.post = nn.Conv1d(
            istftnet_cfg["hidden_dim"],
            dim_out * (1 if disable_complex else 2),
            1,
        )

    def forward(self, x, F0, N, s):
        x = self.pre(x)

        for block in self.blocks:
            x = block(x, s)

        x = self.post(x)

        if self.disable_complex:
            waveform = self.stft.inverse(x, torch.zeros_like(x))
        else:
            real, imag = torch.chunk(x, 2, dim=1)
            magnitude = torch.sqrt(real**2 + imag**2 + 1e-9)
            phase = torch.atan2(imag, real)
            waveform = self.stft.inverse(magnitude, phase)

        return waveform
