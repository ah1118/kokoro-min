from __future__ import annotations

from typing import Dict, List, Optional, Union
import torch

from .model import KModel


# =====================================================
# Helpers
# =====================================================
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


# =====================================================
# Phoneme-only Pipeline (Option 1)
# =====================================================
class KPipeline:
    """
    PHONEMES → AUDIO ONLY

    ❌ No text
    ❌ No G2P
    ❌ No language logic
    ❌ No file I/O
    ❌ No silent truncation

    ✅ Deterministic
    ✅ Modal-friendly
    ✅ RVC-ready
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

        # name -> voice tensor
        self.voices: Dict[str, torch.FloatTensor] = {}

    # -------------------------------------------------
    # Voice handling
    # -------------------------------------------------
    def register_voice(self, name: str, pack: torch.FloatTensor) -> None:
        """
        Register a voice/style embedding tensor.

        pack shape: [N, style_dim]
        """
        self.voices[name] = pack.detach()

    def resolve_voice(self, voice: Union[str, torch.FloatTensor]) -> torch.FloatTensor:
        if isinstance(voice, torch.FloatTensor):
            return voice
        if voice not in self.voices:
            raise ValueError(
                f"Voice '{voice}' not registered. "
                f"Call register_voice(name, tensor) first."
            )
        return self.voices[voice]

    # -------------------------------------------------
    # Inference core
    # -------------------------------------------------
    @staticmethod
    def _infer(
        model: KModel,
        phonemes: str,
        pack: torch.FloatTensor,
        speed: float,
    ) -> torch.FloatTensor:
        """
        Run Kokoro inference and return waveform.
        """
        tokens = phonemes.split()
        if not tokens:
            raise ValueError("Empty phoneme sequence")

        # Style index MUST be based on token count, not characters
        style_index = min(len(tokens) - 1, pack.shape[0] - 1)

        out = model(
            phonemes,
            pack[style_index],
            speed,
            return_output=True,
        )
        return out.audio  # torch.FloatTensor

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def synthesize(
        self,
        phonemes: Union[str, List[str]],
        voice: Union[str, torch.FloatTensor],
        speed: float = 1.0,
    ) -> List[torch.FloatTensor]:
        """
        Synthesize audio from phoneme strings.

        Args:
            phonemes: str or list[str] (PHONEMES ONLY, space-separated)
            voice: registered voice name or tensor
            speed: speech speed multiplier

        Returns:
            List of audio tensors
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

                tokens = ps.split()

                # HARD safety limit (Kokoro context)
                if len(tokens) > 510:
                    raise ValueError(
                        f"Phoneme token length too long ({len(tokens)} > 510). "
                        "Chunk phonemes upstream."
                    )

                audio = self._infer(self.model, ps, pack, speed)
                audios.append(audio)

        return audios


# =====================================================
# Modal-friendly global model cache
# =====================================================
_MODEL: Optional[KModel] = None


def get_model(device: Optional[str] = None) -> KModel:
    """
    Load Kokoro model once per container.
    """
    global _MODEL
    if _MODEL is None:
        dev = _auto_device(device)
        _MODEL = KModel().to(dev).eval()
        _MODEL.requires_grad_(False)
    return _MODEL
