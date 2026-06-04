"""M3 - Translation YO -> EN via NLLB-200-distilled-600M."""
from typing import Any

import torch

import config
from modules.base import PipelineModule
from utils.logging import get_logger

log = get_logger("M3")


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class M3Translate(PipelineModule):
    name = "M3-Translate"

    def __init__(
        self,
        model: str = config.M3_MODEL,
        src_lang: str = config.M3_SRC_LANG,
        tgt_lang: str = config.M3_TGT_LANG,
        max_length: int = 256,
        num_beams: int = 4,
    ) -> None:
        super().__init__()
        self.model_repo = model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = _select_device()
        self.tokenizer = None
        self.model = None
        self._tgt_token_id: int | None = None

    def initialize(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        log.info("loading %s on %s (%s -> %s)",
                 self.model_repo, self.device, self.src_lang, self.tgt_lang)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo, src_lang=self.src_lang
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_repo)
        self.model.to(self.device)
        self.model.eval()
        # NLLB uses convert_tokens_to_ids for the target-language BOS token.
        self._tgt_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        self._ready = True
        log.info("M3 ready (tgt_token_id=%s)", self._tgt_token_id)

    def process(self, input: str) -> dict[str, Any]:
        """Translate diacritized Yoruba (from M2) into English."""
        self._require_ready()
        text = (input or "").strip()
        if not text:
            return {"text": "", "src_lang": self.src_lang, "tgt_lang": self.tgt_lang}

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self._tgt_token_id,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        translated = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        return {
            "text": translated,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "input": text,
        }
