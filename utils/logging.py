"""JSONL logging for intermediate pipeline results (error-propagation analysis)."""
import json
import logging
import time
from pathlib import Path
from typing import Any

import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_stage(stage: str, run_id: str, payload: dict[str, Any]) -> None:
    """Append a JSONL record for one pipeline stage."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LOG_DIR / f"{run_id}.jsonl"
    record = {"stage": stage, "ts": time.time(), **payload}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
