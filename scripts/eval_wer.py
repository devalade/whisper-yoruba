"""WER evaluation for M1 ASR backends on FLEURS yo_ng.

Compares either the mlx-whisper or HuggingFace backend against FLEURS reference
transcripts. Normalization strips diacritics, lowercases, and removes
punctuation — M1's job is non-diacritized text (M2 restores diacritics later),
so judging it on a diacritic-stripped reference is the fair comparison.

Run:
    python -m scripts.eval_wer --asr mlx --n 50
    python -m scripts.eval_wer --asr hf  --n 50 --split validation
    python -m scripts.eval_wer --asr hf  --n 200 --model devalade/whisper-large-v3-yoruba-v3
"""
import argparse
import io
import json
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

import config
from utils.logging import get_logger

log = get_logger("eval")


def strip_diacritics(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def normalize(s: str) -> str:
    s = strip_diacritics(s or "").lower()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(s.split())


def load_asr(backend: str, model: str | None = None, processor: str | None = None):
    if backend == "hf":
        from modules.m1_asr_hf import M1ASRHF
        kwargs: dict[str, str] = {}
        if model:
            kwargs["model"] = model
        if processor:
            kwargs["processor"] = processor
        asr = M1ASRHF(**kwargs)
    else:
        if model or processor:
            log.warning("--model/--processor are only honored by the hf backend; ignoring for mlx")
        from modules.m1_asr import M1ASR
        asr = M1ASR()
    asr.initialize()
    return asr


def iter_samples(split: str, n: int):
    ds = load_dataset("google/fleurs", "yo_ng", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, sample in enumerate(ds):
        if i >= n:
            break
        ref = sample.get("transcription") or sample.get("raw_transcription") or ""
        yield i, sample["audio"], ref


def write_tmp_wav(raw: dict, tmp_dir: Path, name: str) -> Path:
    src = io.BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]
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
    p.add_argument("--asr", choices=["mlx", "hf"], default="mlx",
                   help="which M1 backend to evaluate")
    p.add_argument("--split", default="validation",
                   help="FLEURS split (validation|test)")
    p.add_argument("--n", type=int, default=50,
                   help="max samples to evaluate (Whisper Large is slow on CPU/MPS)")
    p.add_argument("--out", default="logs/wer_results.jsonl")
    p.add_argument("--model", default=None,
                   help="HF model repo to evaluate (hf backend only); "
                        "overrides config.M1_HF_MODEL")
    p.add_argument("--processor", default=None,
                   help="HF processor repo (hf backend only); "
                        "overrides config.M1_HF_PROCESSOR")
    args = p.parse_args()

    asr = load_asr(args.asr, model=args.model, processor=args.processor)
    model_used = getattr(asr, "model_repo", None) or args.model or "(mlx)"

    results: list[dict] = []
    refs: list[str] = []
    hyps: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, raw, ref in iter_samples(args.split, args.n):
            wav = write_tmp_wav(raw, td, f"s_{i:04d}.wav")
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
    print(f"\n=== {args.asr.upper()} [{model_used}] on FLEURS yo_ng/{args.split} (n={len(results)}) ===")
    print(f"Aggregate WER: {agg * 100:.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "backend": args.asr, "model": model_used, "split": args.split,
            "n": len(results), "wer": agg,
        }) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
