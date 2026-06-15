"""Common Voice Yoruba diverse-conditions probe for M1.

Neither v1 nor v3 saw Mozilla Common Voice in training (v1 was Hidi-agili
only; v3's data-mix doc explicitly notes CV was dropped in favor of
chukypedro's broader demographic coverage). That makes CV the only public
Yoruba ASR set this thesis can use for a balanced v1 ↔ v3 A/B on data
neither model trained on.

Honest framing: this is a "diverse real-mic conditions" probe, not a
"spontaneous speech" probe. CV is still read speech, but crowd-recorded
on varied mics/rooms/accents rather than studio-read news prose (FLEURS).
Tests the deployment-distribution risk we actually care about (does v3
regress on noisy/varied input?), not the original convo-eval question
(did v3 fix the conversational segmentation issues?). The latter
requires the unrecorded `data/audio/eval_convo/` set — deferred.

Run:
    # Default = current config.M1_HF_MODEL (v3 after 2026-06-12 flip)
    python -m scripts.cv_probe --n 25 --out logs/cv_probe_v3.jsonl

    # A/B against v1
    python -m scripts.cv_probe --n 25 \
        --model devalade/whisper-large-v3-yoruba \
        --out logs/cv_probe_v1.jsonl

Common Voice is a gated dataset — accept terms at
https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
and `huggingface-cli login` once before running. First load downloads
~hundreds of MB of yo clips into the HF cache.
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

log = get_logger("cv-probe")

DATASET = "mozilla-foundation/common_voice_17_0"
CONFIG = "yo"
DEFAULT_SEED = 42


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
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--split", default="test",
                   help="CV split (validated|test|other). test = validated subset "
                        "held out from train/dev, the cleanest A/B target.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="shuffle seed — keep stable across A/B runs so v1 and v3 "
                        "see the SAME N clips")
    p.add_argument("--model", default=None)
    p.add_argument("--processor", default=None)
    p.add_argument("--out", default="logs/cv_probe.jsonl")
    args = p.parse_args()

    log.info("loading %s [%s] split=%s", DATASET, CONFIG, args.split)
    ds = load_dataset(DATASET, CONFIG, split=args.split)
    total = len(ds)
    log.info("split has %d rows", total)
    if args.n > total:
        log.error("requested n=%d > split size %d", args.n, total)
        sys.exit(2)

    ds = ds.shuffle(seed=args.seed)
    probe = ds.select(range(args.n))
    probe = probe.cast_column("audio", Audio(decode=False))

    asr = load_asr(args.asr, model=args.model, processor=args.processor)
    model_used = getattr(asr, "model_repo", None) or args.model or "(mlx)"

    results: list[dict] = []
    refs: list[str] = []
    hyps: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, row in enumerate(probe):
            wav = _write_wav(row["audio"], td, f"cv_{i:04d}.wav")
            ref = row.get("sentence") or ""
            try:
                hyp = asr.process(wav)["text"]
            except Exception as e:
                log.error("sample %d failed: %s", i, e)
                continue
            ref_n = normalize(ref)
            hyp_n = normalize(hyp)
            wer = jiwer.wer(ref_n, hyp_n) if ref_n else float("nan")
            results.append({
                "i": i, "ref": ref, "hyp": hyp,
                "ref_norm": ref_n, "hyp_norm": hyp_n, "wer": wer,
            })
            refs.append(ref_n)
            hyps.append(hyp_n)
            print(f"[{i:03d}] WER={wer*100:5.1f}% | hyp={hyp[:80]}")

    if not results:
        print("no samples evaluated")
        sys.exit(1)

    agg = jiwer.wer(refs, hyps)
    print(f"\n=== {args.asr.upper()} [{model_used}] on Common Voice yo/{args.split} "
          f"(seed={args.seed}, n={len(results)}) ===")
    print(f"Aggregate WER: {agg * 100:.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "probe": "common-voice-yo",
            "backend": args.asr, "model": model_used,
            "split": args.split, "seed": args.seed,
            "n": len(results), "wer": agg,
        }) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
