"""Fetch one Yoruba audio sample from Google FLEURS (openly hosted) and save as
a 16 kHz mono WAV at data/audio/fleurs_yo_sample.wav. Prints the reference text
so we can compare against the M1 transcript.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

import config

OUT = config.AUDIO_DIR / "fleurs_yo_sample.wav"


def main() -> None:
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print("loading google/fleurs yo_ng[validation] (streaming)…")
    ds = load_dataset("google/fleurs", "yo_ng", split="validation", streaming=True)
    # bypass the (torchcodec-dependent) audio decoder; we'll decode raw bytes ourselves.
    ds = ds.cast_column("audio", Audio(decode=False))
    sample = next(iter(ds))
    raw = sample["audio"]
    # raw is either {"bytes": b"...", "path": "..."} or {"path": "..."} depending on version.
    if raw.get("bytes"):
        arr, sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
    else:
        arr, sr = sf.read(raw["path"], dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)  # downmix to mono
    if sr != config.M1_SAMPLE_RATE:
        # FLEURS is already 16k, but resample defensively.
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=config.M1_SAMPLE_RATE)
        sr = config.M1_SAMPLE_RATE
    sf.write(OUT, arr, sr, subtype="PCM_16")
    print(f"wrote {OUT}  ({len(arr)/sr:.2f}s @ {sr} Hz)")
    print(f"reference transcription: {sample.get('transcription') or sample.get('raw_transcription')}")


if __name__ == "__main__":
    main()
