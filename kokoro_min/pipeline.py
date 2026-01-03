# kokoro_min/pipeline.py
from __future__ import annotations

from typing import Dict, List, Optional, Union
import torch

from .model import KModel


# -----------------------------
# Helpers
# -----------------------------
def _auto_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def _as_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, list):
        return x
    x = x.strip()
    return [x] if x else []


# -----------------------------
# Hardened Phoneme Pipeline
# -----------------------------
class KPipeline:
    """
    Option 1 pipeline: PHONEMES → AUDIO ONLY

    ❌ No text processing
    ❌ No G2P
    ❌ No language logic
    ❌ No file I/O
    ❌ No silent truncation

    ✅ Fast
    ✅ Deterministic
    ✅ Modal-friendly
    """

    def __init__(
        self,
        model: Optional[KModel] = None,
        device: Optional[str] = None,
    ):
        self.device = _auto_device(device)

        self.model = model
        if self.model is not None:
            self.model = self.model.to(self.device).eval()
            self.model.requires_grad_(False)

        # Preloaded voices (name → tensor)
        self.voices: Dict[str, torch.FloatTensor] = {}

    # -------- Voices --------
    def register_voice(self, name: str, pack: torch.FloatTensor) -> None:
        """
        Register a voice embedding pack in memory.
        """
        self.voices[name] = pack.detach()

    def resolve_voice(self, voice: Union[str, torch.FloatTensor]) -> torch.FloatTensor:
        """
        Resolve voice by name or direct tensor.
        """
        if isinstance(voice, torch.FloatTensor):
            return voice
        if voice not in self.voices:
            raise ValueError(
                f"Voice '{voice}' not registered. "
                f"Call register_voice(name, tensor) first."
            )
        return self.voices[voice]

    # -------- Inference core --------
    @staticmethod
    def _infer(
        model: KModel,
        phonemes: str,
        pack: torch.FloatTensor,
        speed: float,
    ) -> torch.FloatTensor:
        """
        Run model inference and return waveform tensor.
        """
        out = model(
            phonemes,
            pack[len(phonemes) - 1],
            speed,
            return_output=True,
        )
        return out.audio  # torch.FloatTensor

    # -------- Public API --------
    def synthesize(
        self,
        phonemes: Union[str, List[str]],
        voice: Union[str, torch.FloatTensor],
        speed: float = 1.0,
    ) -> List[torch.FloatTensor]:
        """
        Synthesize audio from phoneme strings.

        Args:
            phonemes: str or list[str] (PHONEMES ONLY)
            voice: registered voice name or voice tensor
            speed: speech speed multiplier

        Returns:
            List of audio tensors (one per phoneme string)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded in pipeline")
        if voice is None:
            raise ValueError("voice is required")

        segments = _as_list(phonemes)
        if not segments:
            return []

        pack = self.resolve_voice(voice).to(self.device)

        audios: List[torch.FloatTensor] = []

        with torch.inference_mode():
            for ps in segments:
                ps = ps.strip()
                if not ps:
                    continue

                # HARD limit — no silent truncation
                if len(ps) > 510:
                    raise ValueError(
                        f"Phoneme sequence too long ({len(ps)} > 510). "
                        f"Chunk phonemes upstream."
                    )

                audio = self._infer(self.model, ps, pack, speed)
                audios.append(audio)

        return audios


# -----------------------------
# Modal-friendly singleton
# -----------------------------
_MODEL: Optional[KModel] = None


def get_model(device: Optional[str] = None) -> KModel:
    """
    Global cached model (load once per container).
    """
    global _MODEL
    if _MODEL is None:
        dev = _auto_device(device)
        _MODEL = KModel().to(dev).eval()
        _MODEL.requires_grad_(False)
    return _MODEL
