"""M1 - ASR: Whisper Large v3 via mlx-whisper (Apple Silicon)."""
from pathlib import Path
from typing import Any

import config
from modules.base import PipelineModule
from utils.logging import get_logger

log = get_logger("M1")


class M1ASR(PipelineModule):
    name = "M1-ASR"

    def __init__(
        self,
        model: str = config.M1_MODEL,
        language: str = config.M1_LANGUAGE,
    ) -> None:
        super().__init__()
        self.model_repo = model
        self.language = language
        self._transcribe = None

    def initialize(self) -> None:
        """Resolve the mlx-whisper transcribe entrypoint. Weights are lazy-loaded
        by mlx-whisper on first call and cached under ~/.cache/huggingface."""
        log.info("loading mlx-whisper (%s)", self.model_repo)
        import mlx_whisper  # local import so the module can be imported on non-MLX hosts

        self._transcribe = mlx_whisper.transcribe
        self._ready = True
        log.info("M1 ready")

    def process(self, input: str | Path) -> dict[str, Any]:
        """Transcribe a 16 kHz mono WAV file to raw (non-diacritized) Yoruba text.

        Returns a dict with the full text, per-segment timings, and metadata
        downstream modules and the logger can consume.
        """
        self._require_ready()
        audio_path = Path(input)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        log.info("transcribing %s", audio_path.name)
        result = self._transcribe(
            str(audio_path),
            path_or_hf_repo=self.model_repo,
            language=self.language,
            task="transcribe",
            word_timestamps=False,
            condition_on_previous_text=False,
        )

        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        return {
            "text": result["text"].strip(),
            "language": result.get("language", self.language),
            "segments": segments,
            "audio_path": str(audio_path),
        }
