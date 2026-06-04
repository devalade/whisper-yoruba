"""Smoke test for M1. Run: python -m tests.test_m1 path/to/audio.wav"""
import sys
from pathlib import Path

# allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.m1_asr import M1ASR
from utils.logging import log_stage


def main(audio_path: str) -> None:
    run_id = Path(audio_path).stem
    asr = M1ASR()
    asr.initialize()
    out = asr.process(audio_path)
    log_stage("M1", run_id, out)
    print("Transcript:", out["text"])
    print(f"Segments:  {len(out['segments'])}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m tests.test_m1 <audio.wav>")
        sys.exit(1)
    main(sys.argv[1])
