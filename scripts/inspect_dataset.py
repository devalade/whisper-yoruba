"""Sniff-test an HF audio dataset before committing to it for fine-tuning.

Streams a dataset (no full download), pulls a sample of rows, and reports:

- Column schema and dtypes
- Audio sampling rate, dtype, duration distribution
- Distinct speakers in the sample (if a `speaker_id`-like column exists)
- Text length distribution and whether the text appears diacritized
- A handful of clips written to disk so you can actually listen

Designed to answer the question "is this dataset worth spending GPU money
training on?" before any training script touches it.

Usage:
    python -m scripts.inspect_dataset chukypedro/clean_yoruba_dataset
    python -m scripts.inspect_dataset <repo_id> --n 50 --save 10 \\
        --out data/inspect/<name>
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
from datasets import load_dataset

import config

YORUBA_DIACRITIC_CHARS = set("àáèéìíòóùúẹọṣÀÁÈÉÌÍÒÓÙÚẸỌṢ")
SPEAKER_COLUMN_CANDIDATES = ("speaker_id", "speaker", "client_id", "speakerID")
TEXT_COLUMN_CANDIDATES = ("text", "transcription", "sentence", "raw_transcription")
AUDIO_COLUMN_CANDIDATES = ("audio", "wav", "speech")


def pick(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    return None


def has_diacritics(text: str) -> bool:
    if any(ch in YORUBA_DIACRITIC_CHARS for ch in text):
        return True
    # Catch decomposed forms (combining marks)
    return any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", text))


def fmt_dist(values):
    if not values:
        return "n=0"
    return (
        f"n={len(values)} min={min(values):.2f} "
        f"median={median(values):.2f} mean={mean(values):.2f} max={max(values):.2f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_id", help="HF dataset repo id, e.g. chukypedro/clean_yoruba_dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=20, help="rows to materialize from the stream")
    p.add_argument("--save", type=int, default=5, help="clips to write to disk")
    p.add_argument("--out", type=Path, default=None, help="where to write clips (default data/inspect/<name>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip",
        type=int,
        default=0,
        help="skip N rows before sampling — use to probe different regions of large datasets",
    )
    p.add_argument(
        "--shuffle-buffer",
        type=int,
        default=0,
        help="shuffle buffer size for streaming (0 = no shuffle, take sequentially after skip)",
    )
    args = p.parse_args()

    out_dir = args.out or Path("data/inspect") / args.repo_id.replace("/", "__")
    if args.skip:
        out_dir = out_dir.with_name(out_dir.name + f"__skip{args.skip}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.repo_id} (split={args.split}, streaming, skip={args.skip}) ...")
    ds = load_dataset(args.repo_id, split=args.split, streaming=True)
    if args.skip:
        ds = ds.skip(args.skip)
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    rows = []
    it = iter(ds)
    for _ in range(args.n):
        try:
            rows.append(next(it))
        except StopIteration:
            break
    if not rows:
        print("ERROR: dataset returned no rows")
        return 1

    sample = rows[0]
    columns = list(sample.keys())
    print(f"\nColumns: {columns}")
    audio_col = pick(columns, AUDIO_COLUMN_CANDIDATES)
    text_col = pick(columns, TEXT_COLUMN_CANDIDATES)
    speaker_col = pick(columns, SPEAKER_COLUMN_CANDIDATES)
    print(f"  audio column   : {audio_col}")
    print(f"  text column    : {text_col}")
    print(f"  speaker column : {speaker_col}")

    if audio_col is None or text_col is None:
        print("ERROR: could not identify audio or text column — bailing")
        return 2

    durations, srates, dtypes = [], Counter(), Counter()
    text_chars, diacritized = [], 0
    speakers = Counter()
    for r in rows:
        a = r[audio_col]
        arr = np.asarray(a["array"])
        sr = int(a["sampling_rate"])
        durations.append(len(arr) / sr)
        srates[sr] += 1
        dtypes[str(arr.dtype)] += 1
        t = r[text_col] or ""
        text_chars.append(len(t))
        if has_diacritics(t):
            diacritized += 1
        if speaker_col is not None:
            speakers[r.get(speaker_col)] += 1

    print(f"\nAudio (over {len(rows)} sampled rows):")
    print(f"  sampling rates : {dict(srates)}  (target {config.M1_SAMPLE_RATE})")
    print(f"  array dtypes   : {dict(dtypes)}")
    print(f"  duration (s)   : {fmt_dist(durations)}")

    print("\nText:")
    print(f"  length (chars) : {fmt_dist(text_chars)}")
    print(f"  diacritized    : {diacritized}/{len(rows)} rows show Yoruba diacritics")
    print("  examples:")
    for r in rows[:5]:
        snippet = (r[text_col] or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f"    - {snippet!r}")

    if speaker_col is not None:
        top = speakers.most_common(10)
        print(f"\nSpeakers in sample (col `{speaker_col}`):")
        print(f"  distinct       : {len(speakers)} / {len(rows)} rows")
        print(f"  top 10         : {top}")

    n_save = min(args.save, len(rows))
    print(f"\nSaving {n_save} clips to {out_dir}/ for listening:")
    manifest_lines = []
    for i, r in enumerate(rows[:n_save]):
        a = r[audio_col]
        wav_path = out_dir / f"clip_{i:02d}.wav"
        sf.write(wav_path, np.asarray(a["array"]), int(a["sampling_rate"]))
        text = (r[text_col] or "").strip()
        spk = r.get(speaker_col) if speaker_col else None
        manifest_lines.append(f"clip_{i:02d}.wav\t{spk}\t{text}")
        print(f"  {wav_path}")
    (out_dir / "manifest.tsv").write_text(
        "filename\tspeaker\ttext\n" + "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    print(f"  {out_dir/'manifest.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
