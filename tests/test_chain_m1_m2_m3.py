"""End-to-end check: M1 -> M2 -> M3.
Run: python -m tests.test_chain_m1_m2_m3 <audio.wav>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.m1_asr import M1ASR
from modules.m2_diacritic import M2Diacritic
from modules.m3_translate import M3Translate
from utils.logging import log_stage


def main(audio_path: str) -> None:
    run_id = f"chain-{Path(audio_path).stem}"

    m1 = M1ASR(); m1.initialize()
    m2 = M2Diacritic(); m2.initialize()
    m3 = M3Translate(); m3.initialize()

    r1 = m1.process(audio_path); log_stage("M1", run_id, r1)
    r2 = m2.process(r1["text"]); log_stage("M2", run_id, r2)
    r3 = m3.process(r2["text"]); log_stage("M3", run_id, r3)

    print("\n=== M1 (ASR, raw YO) ===")
    print(r1["text"])
    print("\n=== M2 (diacritized YO) ===")
    print(r2["text"])
    print("\n=== M3 (EN) ===")
    print(r3["text"])
    print(f"\n(log: logs/{run_id}.jsonl)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m tests.test_chain_m1_m2_m3 <audio.wav>")
        sys.exit(1)
    main(sys.argv[1])
