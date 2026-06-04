"""M5 - EN -> YO translation + M2 diacritic restoration + MMS-TTS Yoruba.

Closes the loop: takes the English answer from M4 and produces a Yoruba WAV.
"""
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

import config
from modules.base import PipelineModule
from modules.m2_diacritic import M2Diacritic
from utils.logging import get_logger

log = get_logger("M5")


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class M5TTS(PipelineModule):
    name = "M5-TTS"

    def __init__(
        self,
        en_yo_model: str = config.M5_EN_YO_MODEL,
        tts_model: str = config.M5_TTS_MODEL,
        src_lang: str = config.M5_SRC_LANG,
        tgt_lang: str = config.M5_TGT_LANG,
        max_length: int = 256,
        num_beams: int = 4,
        output_dir: Path = config.OUTPUT_DIR,
    ) -> None:
        super().__init__()
        self.en_yo_repo = en_yo_model
        self.tts_repo = tts_model
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.num_beams = num_beams
        self.output_dir = Path(output_dir)
        self.device = _select_device()

        self.tr_tokenizer = None
        self.tr_model = None
        self._tgt_token_id: int | None = None

        # M2 is applied a second time, inside M5, per the architectural spec.
        self.m2 = M2Diacritic()

        self.tts_tokenizer = None
        self.tts_model = None
        self.tts_sr: int | None = None

    def initialize(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel

        log.info("loading EN->YO translator %s on %s", self.en_yo_repo, self.device)
        self.tr_tokenizer = AutoTokenizer.from_pretrained(
            self.en_yo_repo, src_lang=self.src_lang
        )
        self.tr_model = AutoModelForSeq2SeqLM.from_pretrained(self.en_yo_repo)
        self.tr_model.to(self.device); self.tr_model.eval()
        self._tgt_token_id = self.tr_tokenizer.convert_tokens_to_ids(self.tgt_lang)

        log.info("initializing inner M2 (re-diacritization)")
        self.m2.initialize()

        log.info("loading TTS %s on %s", self.tts_repo, self.device)
        self.tts_tokenizer = AutoTokenizer.from_pretrained(self.tts_repo)
        self.tts_model = VitsModel.from_pretrained(self.tts_repo)
        self.tts_model.to(self.device); self.tts_model.eval()
        self.tts_sr = self.tts_model.config.sampling_rate

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ready = True
        log.info("M5 ready (TTS sr=%d)", self.tts_sr)

    def _translate_en_to_yo(self, en_text: str) -> str:
        inputs = self.tr_tokenizer(
            en_text, return_tensors="pt",
            truncation=True, max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            out = self.tr_model.generate(
                **inputs,
                forced_bos_token_id=self._tgt_token_id,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        return self.tr_tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    def _synthesize(self, yo_text: str) -> np.ndarray:
        inputs = self.tts_tokenizer(yo_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.tts_model(**inputs)
        # VitsModel returns .waveform shape (batch, samples); take batch 0 to CPU float32.
        wav = out.waveform[0].detach().to("cpu").float().numpy()
        return wav

    def process(self, input: str, output_path: str | Path | None = None) -> dict[str, Any]:
        """English text in (from M4), Yoruba WAV out. Returns the audio path
        alongside every intermediate string for the error-propagation log."""
        self._require_ready()
        en_text = (input or "").strip()
        if not en_text:
            return {"audio_path": None, "en": "", "yo": "", "yo_diacritized": ""}

        yo_raw = self._translate_en_to_yo(en_text)
        yo_diac = self.m2.process(yo_raw)["text"]
        wav = self._synthesize(yo_diac)

        if output_path is None:
            output_path = self.output_dir / "m5_out.wav"
        output_path = Path(output_path)
        sf.write(output_path, wav, self.tts_sr, subtype="PCM_16")

        return {
            "audio_path": str(output_path),
            "en": en_text,
            "yo": yo_raw,
            "yo_diacritized": yo_diac,
            "sampling_rate": self.tts_sr,
            "duration_s": len(wav) / self.tts_sr,
        }
