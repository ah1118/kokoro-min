from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union
import json
import torch

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder

# --------------------------------------------------
# NO loguru
# --------------------------------------------------
def _log(msg: str):
    # silent by default (safe for Modal)
    return


# --------------------------------------------------
# Model
# --------------------------------------------------
class KModel(torch.nn.Module):
    """
    Minimal Kokoro model (PHONEMES → AUDIO)

    ❌ No HF auto-download
    ❌ No logging deps
    ❌ No magic paths

    ✅ Explicit
    ✅ Deterministic
    """

    def __init__(
        self,
        *,
        config_path: str,
        model_path: str,
        disable_complex: bool = False,
    ):
        super().__init__()

        # ---------------------------
        # Load config
        # ---------------------------
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.vocab: Dict[str, int] = config["vocab"]

        # ---------------------------
        # Lazy import (transformers)
        # ---------------------------
        try:
            from transformers import AlbertConfig
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for kokoro-min model.py\n"
                "Install it explicitly in your image:\n"
                "pip install transformers"
            ) from e

        # ---------------------------
        # Build network
        # ---------------------------
        self.bert = CustomAlbert(
            AlbertConfig(
                vocab_size=config["n_token"],
                **config["plbert"],
            )
        )

        self.bert_encoder = torch.nn.Linear(
            self.bert.config.hidden_size,
            config["hidden_dim"],
        )

        self.context_length = self.bert.config.max_position_embeddings

        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )

        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )

        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            disable_complex=disable_complex,
            **config["istftnet"],
        )

        # ---------------------------
        # Load weights (EXPLICIT)
        # ---------------------------
        state = torch.load(model_path, map_location="cpu", weights_only=True)

        for key, sd in state.items():
            if not hasattr(self, key):
                continue
            try:
                getattr(self, key).load_state_dict(sd)
            except RuntimeError:
                # tolerate "module." prefix mismatch
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                getattr(self, key).load_state_dict(sd, strict=False)

        self.eval()
        self.requires_grad_(False)

    # --------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    # --------------------------------------------------
    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    # --------------------------------------------------
    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
    ):
        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=input_ids.device,
            dtype=torch.long,
        )

        text_mask = (
            torch.arange(input_lengths.max(), device=self.device)
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
        )
        text_mask = (text_mask + 1 > input_lengths.unsqueeze(1))

        bert_out = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_out).transpose(-1, -2)

        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        idx = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=self.device),
            pred_dur,
        )

        aln = torch.zeros(
            (input_ids.shape[1], idx.shape[0]),
            device=self.device,
        )
        aln[idx, torch.arange(idx.shape[0])] = 1
        aln = aln.unsqueeze(0)

        en = d.transpose(-1, -2) @ aln
        F0, N = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln

        audio = self.decoder(asr, F0, N, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    # --------------------------------------------------
    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_output: bool = False,
    ):
        tokens = phonemes.split()
        ids = [self.vocab.get(t) for t in tokens if t in self.vocab]

        if len(ids) + 2 > self.context_length:
            raise ValueError(
                f"Too many phonemes ({len(ids)} > {self.context_length})"
            )

        input_ids = torch.LongTensor([[0, *ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)

        audio, pred_dur = self.forward_with_tokens(
            input_ids, ref_s, speed
        )

        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None

        if return_output:
            return self.Output(audio=audio, pred_dur=pred_dur)
        return audio
