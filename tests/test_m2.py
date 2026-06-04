"""Smoke test for M2. Run: python -m tests.test_m2 "raw yoruba text without diacritics" """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.m2_diacritic import M2Diacritic
from utils.logging import log_stage


def main(text: str) -> None:
    m2 = M2Diacritic()
    m2.initialize()
    out = m2.process(text)
    log_stage("M2", "m2-smoke", {"input": text, **out})
    print("Input:    ", text)
    print("Restored: ", out["text"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: python -m tests.test_m2 "bawo ni o se wa"')
        sys.exit(1)
    main(sys.argv[1])
