"""M1 alternate - HuggingFace Whisper backend.

Drop-in replacement for `M1ASR` (returns the same dict shape) that loads a
Yoruba-fine-tuned Whisper Large v2 checkpoint via `transformers`. Device and
attention impl are auto-selected for the host (MPS / CUDA / CPU); flash-attn
is used only when actually available.
"""
from pathlib import Path
from typing import Any

import torch

import config
from modules.base import PipelineModule
from utils.logging import get_logger

log = get_logger("M1-hf")


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str) -> torch.dtype:
    # fp16 on CUDA is the snippet's intent; on MPS fp16 Whisper occasionally
    # produces NaNs in generate(), so stay in fp32 there. CPU stays fp32.
    return torch.float16 if device == "cuda" else torch.float32


def _select_attn_impl(device: str) -> str:
    if device == "cuda":
        try:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                return "flash_attention_2"
        except Exception:
            pass
    return "sdpa"


class M1ASRHF(PipelineModule):
    name = "M1-ASR-HF"

    def __init__(
        self,
        model: str = config.M1_HF_MODEL,
        processor: str = config.M1_HF_PROCESSOR,
        language: str = config.M1_LANGUAGE,
        chunk_length_s: float = config.M1_HF_CHUNK_S,
        num_beams: int = config.M1_HF_NUM_BEAMS,
        max_new_tokens: int = config.M1_HF_MAX_NEW_TOKENS,
    ) -> None:
        super().__init__()
        self.model_repo = model
        self.processor_repo = processor
        self.language = language
        self.chunk_length_s = chunk_length_s
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.device = _select_device()
        self.dtype = _select_dtype(self.device)
        self.attn_impl = _select_attn_impl(self.device)
        self.pipe = None

    def initialize(self) -> None:
        from transformers import (
            WhisperForConditionalGeneration,
            pipeline,
        )

        log.info(
            "loading %s on %s (dtype=%s, attn=%s)",
            self.model_repo, self.device, self.dtype, self.attn_impl,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_repo,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        ).to(self.device)

        # The fine-tune ships forced_decoder_ids in generation_config; the
        # snippet moves them to input_ids to match the new generate() contract.
        if getattr(model.generation_config, "forced_decoder_ids", None) is not None:
            model.generation_config.input_ids = model.generation_config.forced_decoder_ids
            model.generation_config.forced_decoder_ids = None

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            processor=self.processor_repo,
            tokenizer=self.processor_repo,
            feature_extractor=self.processor_repo,
            chunk_length_s=self.chunk_length_s,
            device=self.device,
            generate_kwargs={
                "num_beams": self.num_beams,
                "max_new_tokens": self.max_new_tokens,
                "early_stopping": True,
                "language": self.language,
                "task": "transcribe",
            },
        )
        self._ready = True
        log.info("M1-HF ready")

    def process(self, input: str | Path) -> dict[str, Any]:
        self._require_ready()
        audio_path = Path(input)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        log.info("transcribing %s", audio_path.name)
        out = self.pipe(str(audio_path))
        text = (out.get("text") or "").strip()
        # The HF pipeline doesn't return per-segment timings by default; keep
        # the shape compatible with M1ASR so downstream/logging stays uniform.
        return {
            "text": text,
            "language": self.language,
            "segments": [],
            "audio_path": str(audio_path),
        }
