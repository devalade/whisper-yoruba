"""Record the conversational eval set, one phrase at a time.

Walks `data/eval/convo_phrases.jsonl`, prompts you to record each phrase via
push-to-talk, and writes 16 kHz mono WAVs to `data/audio/eval_convo/<id>.wav`.

Already-recorded phrases are skipped by default — re-record one with
`--rerecord <id>` (repeatable) or every phrase with `--rerecord-all`.

Run:
    python -m scripts.record_convo
    python -m scripts.record_convo --rerecord ident_01 --rerecord ident_03
    python -m scripts.record_convo --rerecord-all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from utils.mic import record_push_to_talk

PHRASES = config.DATA_DIR / "eval" / "convo_phrases.jsonl"
OUT_DIR = config.AUDIO_DIR / "eval_convo"


def load_phrases() -> list[dict]:
    with PHRASES.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rerecord", action="append", default=[],
                   help="phrase id(s) to re-record (repeat flag)")
    p.add_argument("--rerecord-all", action="store_true",
                   help="re-record every phrase, overwriting existing WAVs")
    args = p.parse_args()

    phrases = load_phrases()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    force = set(args.rerecord)

    print(f"\n{len(phrases)} phrases. Press Enter to start each one, Enter again to stop.")
    print(f"WAVs go to {OUT_DIR.relative_to(config.ROOT)}/\n")

    for i, ph in enumerate(phrases, 1):
        out = OUT_DIR / f"{ph['id']}.wav"
        if out.exists() and not args.rerecord_all and ph["id"] not in force:
            print(f"[{i:02d}/{len(phrases)}] {ph['id']:10s}  (skip, already recorded)")
            continue
        print(f"\n[{i:02d}/{len(phrases)}] {ph['id']:10s}  [{ph['category']}]")
        print(f"  YO:  {ph['yo']}")
        print(f"  EN:  {ph['en']}")
        try:
            record_push_to_talk(out)
        except KeyboardInterrupt:
            print("\nquit — partial set saved.")
            return
        except RuntimeError as e:
            print(f"  failed: {e} — moving on")
            continue

    print("\ndone.")


if __name__ == "__main__":
    main()
