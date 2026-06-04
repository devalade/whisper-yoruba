"""Smoke test for M4. Tests an in-domain query and an out-of-domain query
(to verify the similarity-threshold refusal).
Run: python -m tests.test_m4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.m4_rag import M4RAG
from utils.logging import log_stage

QUERIES = [
    "Who is Wole Soyinka?",
    "What is the capital of France?",  # out-of-domain — should refuse
]


def main() -> None:
    m4 = M4RAG()
    m4.initialize()
    for i, q in enumerate(QUERIES):
        out = m4.process(q)
        log_stage("M4", f"m4-smoke-{i}", out)
        print("\n--- Q:", q)
        print(f"max_sim={out['max_sim']:.3f}  below_threshold={out['below_threshold']}")
        print("A:", out["answer"])


if __name__ == "__main__":
    main()
