"""Push-to-talk mic capture + playback helpers for --mic mode."""
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

import config


def record_push_to_talk(out_path: str | Path, sample_rate: int = config.M1_SAMPLE_RATE) -> Path:
    """Press Enter to start, Enter to stop. Writes a 16 kHz mono WAV.

    Type "q" + Enter (or Ctrl+D) at the start prompt to quit — raises
    KeyboardInterrupt so the caller's loop exits cleanly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ans = input("Press Enter to speak (or 'q' + Enter to quit)… ")
    except EOFError:
        raise KeyboardInterrupt
    if ans.strip().lower() in {"q", "quit", "exit"}:
        raise KeyboardInterrupt

    print("Recording. Press Enter to stop.")

    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"[mic] {status}")
        frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()

    if not frames:
        raise RuntimeError("no audio captured")

    audio = np.concatenate(frames, axis=0).squeeze()
    sf.write(out_path, audio, sample_rate, subtype="PCM_16")
    print(f"saved: {out_path}  ({len(audio) / sample_rate:.2f}s)")
    return out_path


def play_wav(path: str | Path) -> None:
    audio, sr = sf.read(str(path), dtype="float32")
    print(f"playing: {path}")
    sd.play(audio, sr)
    sd.wait()
