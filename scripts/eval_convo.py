"""WER eval for M1 on the conversational eval set.

Runs the chosen M1 backend on every recorded WAV in `data/audio/eval_convo/`,
matches it against the diacritized reference from `data/eval/convo_phrases.jsonl`,
and reports WER overall + per category. Normalization matches scripts/eval_wer.py
(strip diacritics, lowercase, collapse punctuation) — fair comparison against
FLEURS numbers.

Run:
    python -m scripts.eval_convo --asr hf
    python -m scripts.eval_convo --asr mlx --out logs/wer_convo_mlx.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer

import config
from scripts.eval_wer import load_asr, normalize
from utils.logging import get_logger

log = get_logger("eval-convo")

PHRASES = config.DATA_DIR / "eval" / "convo_phrases.jsonl"
AUDIO_DIR = config.AUDIO_DIR / "eval_convo"


def load_phrases() -> list[dict]:
    with PHRASES.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asr", choices=["mlx", "hf"], default="hf")
    p.add_argument("--out", default=None,
                   help="JSONL output path (default logs/wer_convo_<backend>.jsonl)")
    args = p.parse_args()

    out_path = Path(args.out or f"logs/wer_convo_{args.asr}.jsonl")
    phrases = load_phrases()
    asr = load_asr(args.asr)

    results: list[dict] = []
    by_cat: dict[str, list[tuple[str, str]]] = defaultdict(list)
    refs: list[str] = []
    hyps: list[str] = []
    missing: list[str] = []

    for ph in phrases:
        wav = AUDIO_DIR / f"{ph['id']}.wav"
        if not wav.exists():
            missing.append(ph["id"])
            continue
        try:
            hyp = asr.process(wav)["text"]
        except Exception as e:
            log.error("%s failed: %s", ph["id"], e)
            continue
        ref_n = normalize(ph["yo"])
        hyp_n = normalize(hyp)
        wer = jiwer.wer(ref_n, hyp_n) if ref_n else float("nan")
        results.append({
            "id": ph["id"], "category": ph["category"],
            "ref": ph["yo"], "hyp": hyp,
            "ref_norm": ref_n, "hyp_norm": hyp_n, "wer": wer,
        })
        by_cat[ph["category"]].append((ref_n, hyp_n))
        refs.append(ref_n)
        hyps.append(hyp_n)
        mark = "✓" if wer == 0 else " "
        print(f"  {mark} {ph['id']:10s} WER={wer*100:5.1f}%  ref={ph['yo']!r}  hyp={hyp.strip()!r}")

    if missing:
        print(f"\n[warn] {len(missing)} phrase(s) not recorded yet: {', '.join(missing)}")
        print("       run: python -m scripts.record_convo")

    if not results:
        print("no recordings to evaluate")
        sys.exit(1)

    print(f"\n=== {args.asr.upper()} on convo eval set (n={len(results)}) ===")
    for cat in sorted(by_cat):
        cat_refs = [r for r, _ in by_cat[cat]]
        cat_hyps = [h for _, h in by_cat[cat]]
        cat_wer = jiwer.wer(cat_refs, cat_hyps)
        print(f"  {cat:14s} n={len(cat_refs):2d}  WER={cat_wer*100:5.1f}%")
    agg = jiwer.wer(refs, hyps)
    print(f"  {'OVERALL':14s} n={len(refs):2d}  WER={agg*100:5.1f}%")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "backend": args.asr, "n": len(results),
            "wer": agg,
            "by_category": {c: jiwer.wer([r for r, _ in v], [h for _, h in v])
                            for c, v in by_cat.items()},
            "missing": missing,
        }) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
