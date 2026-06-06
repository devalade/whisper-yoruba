"""Test the fine-tuned M1 on the held-out audio samples.

Reads `data/audio/holdout/manifest.jsonl`, runs the chosen M1 backend on each
WAV, prints reference vs hypothesis side by side, and reports diacritic-stripped
WER. The manifest is produced by `scripts/finetune_whisper.py --holdout-n N`.

The point: these clips were NOT seen during training or eval — they're a clean
sanity check that the model generalizes on data from the training distribution.

Run:
    # After scp'ing the holdout/ dir back from the GPU box:
    python -m scripts.test_holdout --asr hf
    python -m scripts.test_holdout --asr mlx       # baseline for comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer

import config
from scripts.eval_wer import load_asr, normalize

MANIFEST = config.AUDIO_DIR / "holdout" / "manifest.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asr", choices=["mlx", "hf"], default="hf")
    p.add_argument("--manifest", default=str(MANIFEST))
    args = p.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"no manifest at {manifest} — run the training script with "
              f"--holdout-n N and scp the holdout/ dir back from the GPU box.")
        sys.exit(1)

    with manifest.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    asr = load_asr(args.asr)

    refs, hyps = [], []
    for row in rows:
        wav = config.ROOT / row["wav"]
        if not wav.exists():
            print(f"  [missing] {row['id']} — expected at {wav}")
            continue
        hyp = asr.process(wav)["text"].strip()
        ref_n = normalize(row["yo"])
        hyp_n = normalize(hyp)
        wer = jiwer.wer(ref_n, hyp_n) if ref_n else float("nan")
        mark = "✓" if wer == 0 else " "
        print(f"\n{mark} {row['id']}  ({row['duration_s']}s)  WER={wer*100:.1f}%")
        print(f"  REF: {row['yo']}")
        print(f"  HYP: {hyp}")
        refs.append(ref_n)
        hyps.append(hyp_n)

    if refs:
        agg = jiwer.wer(refs, hyps)
        print(f"\n=== {args.asr.upper()} on holdout (n={len(refs)}) ===")
        print(f"Aggregate WER: {agg * 100:.2f}%")


if __name__ == "__main__":
    main()
