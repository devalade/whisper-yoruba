"""Merge a Whisper LoRA adapter into the base model.

After `scripts/finetune_whisper.py` saves an adapter to `config.FT_OUTPUT_DIR`,
this script loads the base + adapter, merges weights, and writes a plain
WhisperForConditionalGeneration checkpoint to `config.FT_MERGED_DIR`. The
merged checkpoint loads as a regular Whisper model — `modules/m1_asr_hf.py`
needs no code changes, just point `config.M1_HF_MODEL` at the merged dir or
the pushed HF repo.

Usage:
    python -m scripts.merge_lora
    python -m scripts.merge_lora --push-to-hub --hub-model-id <user>/whisper-large-v3-yoruba
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import config
from utils.logging import get_logger

log = get_logger("merge-lora")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=config.FT_BASE_MODEL)
    p.add_argument("--adapter-dir", default=str(config.FT_OUTPUT_DIR))
    p.add_argument("--out-dir", default=str(config.FT_MERGED_DIR))
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("loading base %s", args.base_model)
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    log.info("loading adapter %s", args.adapter_dir)
    merged = PeftModel.from_pretrained(base, args.adapter_dir)

    log.info("merging adapter into base weights")
    merged = merged.merge_and_unload()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("saving merged model to %s", out)
    merged.save_pretrained(out)

    # Save the processor alongside so inference can load it from the same dir
    # without touching the base repo.
    processor = WhisperProcessor.from_pretrained(args.adapter_dir)
    processor.save_pretrained(out)

    if args.push_to_hub:
        if not args.hub_model_id:
            raise SystemExit("--push-to-hub requires --hub-model-id")
        log.info("pushing to %s", args.hub_model_id)
        merged.push_to_hub(args.hub_model_id)
        processor.push_to_hub(args.hub_model_id)

    log.info(
        "done — set config.M1_HF_MODEL = %r (or your hub id) to use this checkpoint",
        str(out),
    )


if __name__ == "__main__":
    main()
