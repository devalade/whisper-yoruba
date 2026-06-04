"""M4 - free-form chat: prompts the local Mistral directly, no retrieval.

Drop-in replacement for M4RAG. Same return shape so pipeline.py and the M5 leg
don't change. Use when you want the model to answer from its own parametric
knowledge instead of from the indexed Wikipedia corpus.
"""
from pathlib import Path
from typing import Any

import config
from modules.base import PipelineModule
from utils.logging import get_logger

log = get_logger("M4-chat")

PROMPT_TEMPLATE = """[INST] You are a helpful assistant answering a question \
from a Yoruba speaker. Answer concisely in plain English (2-4 sentences). \
If you genuinely don't know, say so.

Question: {question} [/INST]"""


class M4Chat(PipelineModule):
    name = "M4-chat"

    def __init__(
        self,
        llm_path: Path = config.M4_LLM_PATH,
        n_ctx: int = 4096,
        max_tokens: int = 256,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.llm_path = Path(llm_path)
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm = None

    def initialize(self) -> None:
        from llama_cpp import Llama

        if not self.llm_path.exists():
            raise FileNotFoundError(f"LLM weights not found at {self.llm_path}")

        log.info("loading LLM %s", self.llm_path.name)
        self.llm = Llama(
            model_path=str(self.llm_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        self._ready = True
        log.info("M4-chat ready")

    def process(self, input: str) -> dict[str, Any]:
        self._require_ready()
        question = (input or "").strip()
        if not question:
            return {"answer": "", "passages": [], "max_sim": 0.0,
                    "below_threshold": False, "question": ""}

        prompt = PROMPT_TEMPLATE.format(question=question)
        out = self.llm(prompt, max_tokens=self.max_tokens,
                       temperature=self.temperature, stop=["</s>", "[INST]"])
        answer = out["choices"][0]["text"].strip()
        return {
            "answer": answer,
            "passages": [],     # no retrieval
            "max_sim": 0.0,     # not applicable
            "below_threshold": False,
            "question": question,
        }
