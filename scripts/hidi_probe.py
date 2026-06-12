"""Hidi-agili distribution probe for the current M1 HF backend.

Not a true held-out set. The v3 trial doc cites `make test-holdout` as a
follow-up to confirm v3 didn't catastrophically forget the Hidi-agili TTS
voice, but no holdout WAVs were produced for v3 (`make finetune` doesn't
pass `--holdout-n`), and the GPU box that ran v3 is destroyed.

The next-best signal: load `Hidi-agili/yoruba_tts_dataset`, apply the
same `shuffle(seed=SEED)` the trainer used, then select rows BEYOND v3's
5,000-row training slice. v3 did not see these specific rows. Same
dataset, same speaker, same audio characteristics — so a v3 WER in line
with FLEURS-protocol expectations (~60-70% range) means the anchor held;
a sharp regression (e.g. 90%+) means v3 forgot the TTS distribution.

Run:
    python -m scripts.hidi_probe --n 25
    python -m scripts.hidi_probe --n 25 --model devalade/whisper-large-v3-yoruba
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

import config
from scripts.eval_wer import load_asr, normalize
from utils.logging import get_logger

log = get_logger("hidi-probe")

DATASET = "Hidi-agili/yoruba_tts_dataset"
V3_TRAIN_CAP = 5000  # config.FT_DATA_MIX[0]["max_rows"]
V3_SEED = 42         # scripts/finetune_whisper.py --seed default


def _write_wav(audio: dict, tmp_dir: Path, name: str) -> Path:
    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    data, sr = sf.read(src)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != config.M1_SAMPLE_RATE:
        import librosa
        data = librosa.resample(
            data.astype(np.float32), orig_sr=sr, target_sr=config.M1_SAMPLE_RATE
        )
        sr = config.M1_SAMPLE_RATE
    out = tmp_dir / name
    sf.write(out, data, sr, subtype="PCM_16")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asr", choices=["hf", "mlx"], default="hf")
    p.add_argument("--n", type=int, default=25,
                   help="rows to probe from the unseen-by-v3 slice")
    p.add_argument("--seed", type=int, default=V3_SEED,
                   help=f"shuffle seed — must match training (default {V3_SEED})")
    p.add_argument("--skip", type=int, default=V3_TRAIN_CAP,
                   help=f"rows to skip (default {V3_TRAIN_CAP} = v3's training cap)")
    p.add_argument("--model", default=None,
                   help="override config.M1_HF_MODEL for this run")
    p.add_argument("--processor", default=None,
                   help="override config.M1_HF_PROCESSOR for this run")
    p.add_argument("--out", default="logs/hidi_probe.jsonl")
    args = p.parse_args()

    log.info("loading %s (split=train)", DATASET)
    ds = load_dataset(DATASET, split="train")
    total = len(ds)
    log.info("dataset has %d rows", total)
    if args.skip + args.n > total:
        log.error("skip=%d + n=%d exceeds dataset size %d", args.skip, args.n, total)
        sys.exit(2)

    ds = ds.shuffle(seed=args.seed)
    probe = ds.select(range(args.skip, args.skip + args.n))
    probe = probe.cast_column("audio", Audio(decode=False))

    asr = load_asr(args.asr, model=args.model, processor=args.processor)
    model_used = getattr(asr, "model_repo", None) or args.model or "(mlx)"

    results: list[dict] = []
    refs: list[str] = []
    hyps: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, row in enumerate(probe):
            wav = _write_wav(row["audio"], td, f"hidi_{i:04d}.wav")
            ref = row.get("text") or row.get("transcription") or ""
            try:
                hyp = asr.process(wav)["text"]
            except Exception as e:
                log.error("sample %d failed: %s", i, e)
                continue
            ref_n = normalize(ref)
            hyp_n = normalize(hyp)
            wer = jiwer.wer(ref_n, hyp_n) if ref_n else float("nan")
            results.append({
                "i": args.skip + i, "ref": ref, "hyp": hyp,
                "ref_norm": ref_n, "hyp_norm": hyp_n, "wer": wer,
            })
            refs.append(ref_n)
            hyps.append(hyp_n)
            print(f"[{i:03d}] WER={wer*100:5.1f}% | hyp={hyp[:80]}")

    if not results:
        print("no samples evaluated")
        sys.exit(1)

    agg = jiwer.wer(refs, hyps)
    print(f"\n=== {args.asr.upper()} [{model_used}] on Hidi-agili unseen "
          f"(seed={args.seed}, skip={args.skip}, n={len(results)}) ===")
    print(f"Aggregate WER: {agg * 100:.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "probe": "hidi-agili-unseen",
            "backend": args.asr, "model": model_used,
            "seed": args.seed, "skip": args.skip,
            "n": len(results), "wer": agg,
        }) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
