"""M2 - Diacritic restoration for Yoruba (HuggingFace seq2seq).

Used twice in the pipeline:
  1. After M1, before YO -> EN translation.
  2. Inside M5, after EN -> YO translation, before TTS.
"""
from typing import Any

import torch

import config
from modules.base import PipelineModule
from utils.logging import get_logger

log = get_logger("M2")


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class M2Diacritic(PipelineModule):
    name = "M2-Diacritic"

    def __init__(
        self,
        model: str = config.M2_MODEL_ORIFE,
        max_input_tokens: int = 256,
        max_output_tokens: int = 256,
        num_beams: int = 4,
    ) -> None:
        super().__init__()
        self.model_repo = model
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.num_beams = num_beams
        self.device = _select_device()
        self.tokenizer = None
        self.model = None

    def initialize(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        log.info("loading %s on %s", self.model_repo, self.device)
        # use_fast=False keeps SentencePiece byte-fallback (avoids <unk> on rare
        # Yoruba characters); legacy=False opts into the modern T5 tokenization.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo, use_fast=False, legacy=False
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_repo)
        self.model.to(self.device)
        self.model.eval()
        self._ready = True
        log.info("M2 ready")

    def _restore_one(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=self.max_output_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _split_sentences(self, text: str) -> list[str]:
        # cheap sentence split; keeps punctuation attached to the previous chunk.
        out, buf = [], []
        for tok in text.split():
            buf.append(tok)
            if tok.endswith((".", "?", "!")):
                out.append(" ".join(buf))
                buf = []
        if buf:
            out.append(" ".join(buf))
        return out or [text]

    def process(self, input: str) -> dict[str, Any]:
        """Restore diacritics. Long inputs are split per-sentence to stay within
        the model's input window without losing context boundaries."""
        self._require_ready()
        if not input or not input.strip():
            return {"text": "", "chunks": []}

        chunks = self._split_sentences(input.strip())
        restored = [self._restore_one(c) for c in chunks]
        return {
            "text": " ".join(restored),
            "chunks": [
                {"raw": r, "restored": d} for r, d in zip(chunks, restored)
            ],
        }
