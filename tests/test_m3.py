"""Smoke test for M3. Run: python -m tests.test_m3 "diacritized yoruba sentence" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.m3_translate import M3Translate
from utils.logging import log_stage


def main(text: str) -> None:
    m3 = M3Translate()
    m3.initialize()
    out = m3.process(text)
    log_stage("M3", "m3-smoke", out)
    print("YO in :", text)
    print("EN out:", out["text"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: python -m tests.test_m3 "báwo ni ó ṣe wá lónìí"')
        sys.exit(1)
    main(sys.argv[1])
