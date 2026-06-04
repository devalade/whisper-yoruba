"""Smoke test for M5. Run: python -m tests.test_m5 "English text to speak as Yoruba" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.m5_tts import M5TTS
from utils.logging import log_stage


def main(text: str) -> None:
    m5 = M5TTS()
    m5.initialize()
    out = m5.process(text, output_path="data/outputs/m5_smoke.wav")
    log_stage("M5", "m5-smoke", out)
    print("EN  :", out["en"])
    print("YO  :", out["yo"])
    print("YO+ :", out["yo_diacritized"])
    print(f"WAV : {out['audio_path']}  ({out['duration_s']:.2f}s @ {out['sampling_rate']} Hz)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: python -m tests.test_m5 "Lagos is the largest city in Nigeria."')
        sys.exit(1)
    main(sys.argv[1])
