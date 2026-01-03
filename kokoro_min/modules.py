import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import weight_norm

# --------------------------------------------------
# Explicit transformers dependency
# --------------------------------------------------
try:
    from transformers import AlbertModel
except ImportError as e:
    raise RuntimeError(
        "transformers is required for kokoro-min.\n"
        "Install it explicitly:\n"
        "pip install transformers"
    ) from e


# ==================================================
# Utility layers
# ==================================================
class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear.weight,
            gain=nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x):
        return self.linear(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


# ==================================================
# Text Encoder
# ==================================================
class TextEncoder(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        depth,
        n_symbols,
        actv=nn.LeakyReLU(0.2),
    ):
        super().__init__()

        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            padding=padding,
                        )
                    ),
                    LayerNorm(channels),
                    actv,
                    nn.Dropout(0.2),
                )
                for _ in range(depth)
            ]
        )

        self.lstm = nn.LSTM(
            channels,
            channels // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths, mask):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        mask = mask.unsqueeze(1)
        x.masked_fill_(mask, 0.0)

        for block in self.cnn:
            x = block(x)
            x.masked_fill_(mask, 0.0)

        x = x.transpose(1, 2)

        lengths = (
            input_lengths
            if input_lengths.device.type == "cpu"
            else input_lengths.cpu()
        )

        x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(
            x,
            batch_first=True,
        )

        x = x.transpose(-1, -2)

        x_pad = torch.zeros(
            (x.shape[0], x.shape[1], mask.shape[-1]),
            device=x.device,
        )
        x_pad[:, :, : x.shape[-1]] = x
        x = x_pad

        x.masked_fill_(mask, 0.0)
        return x


# ==================================================
# AdaIN blocks
# ==================================================
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s).unsqueeze(-1)
        gamma, beta = torch.chunk(h, 2, dim=1)
        gamma = gamma.transpose(1, -1)
        beta = beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta

        return x.transpose(1, -1).transpose(-1, -2)


# ==================================================
# Duration / Prosody
# ==================================================
class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(nlayers):
            self.blocks.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.blocks.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout

    def forward(self, x, style, lengths, mask):
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)

        x.masked_fill_(mask.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1).transpose(-1, -2)

        for block in self.blocks:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
                x.masked_fill_(mask.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths_cpu = (
                    lengths if lengths.device.type == "cpu" else lengths.cpu()
                )
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(
                    x,
                    lengths_cpu,
                    batch_first=True,
                    enforce_sorted=False,
                )
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    x,
                    batch_first=True,
                )
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)

                pad = torch.zeros(
                    (x.shape[0], x.shape[1], mask.shape[-1]),
                    device=x.device,
                )
                pad[:, :, : x.shape[-1]] = x
                x = pad

        return x.transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(
        self,
        style_dim,
        d_hid,
        nlayers,
        max_dur=50,
        dropout=0.1,
    ):
        super().__init__()

        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )

        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1)

    def forward(self, texts, style, lengths, alignment, mask):
        d = self.text_encoder(texts, style, lengths, mask)

        lengths_cpu = lengths if lengths.device.type == "cpu" else lengths.cpu()

        x = nn.utils.rnn.pack_padded_sequence(
            d,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x,
            batch_first=True,
        )

        pad = torch.zeros(
            (x.shape[0], mask.shape[-1], x.shape[-1]),
            device=x.device,
        )
        pad[:, : x.shape[1], :] = x
        x = pad

        duration = self.duration_proj(F.dropout(x, 0.5, training=False))

        en = d.transpose(-1, -2) @ alignment
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        x = x.transpose(-1, -2)
        return self.F0_proj(x).squeeze(1), self.N_proj(x).squeeze(1)


# ==================================================
# Custom Albert wrapper
# ==================================================
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return out.last_hidden_state
