"""M4 - RAG: FAISS (IndexFlatIP) + MiniLM-L6-v2 + Mistral 7B Q4 (llama.cpp)."""
import json
from pathlib import Path
from typing import Any

import config
from modules.base import PipelineModule
from utils.logging import get_logger

log = get_logger("M4")

INDEX_PATH = config.WIKI_DIR / "faiss.index"
META_PATH = config.WIKI_DIR / "passages.jsonl"

# llama.cpp adds the BOS token automatically, so the template starts at [INST].
PROMPT_TEMPLATE = """[INST] You answer questions strictly from the provided context. \
If the answer is not contained in the context, reply exactly: "I don't know based on the provided context."

Context:
{context}

Question: {question} [/INST]"""

NO_ANSWER = "I don't know based on the provided context."


class M4RAG(PipelineModule):
    name = "M4-RAG"

    def __init__(
        self,
        embed_model: str = config.M4_EMBED_MODEL,
        llm_path: Path = config.M4_LLM_PATH,
        top_k: int = config.M4_TOP_K,
        sim_threshold: float = config.M4_SIM_THRESHOLD,
        n_ctx: int = 4096,
        max_tokens: int = 256,
    ) -> None:
        super().__init__()
        self.embed_model_id = embed_model
        self.llm_path = Path(llm_path)
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.embedder = None
        self.index = None
        self.passages: list[dict] = []
        self.llm = None

    def initialize(self) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer
        from llama_cpp import Llama

        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run: python -m scripts.build_index"
            )
        if not self.llm_path.exists():
            raise FileNotFoundError(f"LLM weights not found at {self.llm_path}")

        log.info("loading embedder %s", self.embed_model_id)
        self.embedder = SentenceTransformer(self.embed_model_id)

        log.info("loading FAISS index %s", INDEX_PATH)
        self.index = faiss.read_index(str(INDEX_PATH))
        with META_PATH.open(encoding="utf-8") as f:
            self.passages = [json.loads(l) for l in f]
        log.info("loaded %d passages", len(self.passages))

        log.info("loading LLM %s", self.llm_path.name)
        self.llm = Llama(
            model_path=str(self.llm_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,   # offload all layers to Metal on Apple Silicon
            verbose=False,
        )
        self._ready = True
        log.info("M4 ready")

    def _search(self, query: str) -> tuple[float, list[dict]]:
        import numpy as np

        vec = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        sims, idxs = self.index.search(vec, self.top_k)
        sims, idxs = sims[0].tolist(), idxs[0].tolist()
        hits = [
            {**self.passages[i], "sim": float(s)}
            for s, i in zip(sims, idxs) if i != -1
        ]
        max_sim = max(sims) if sims else 0.0
        return max_sim, hits

    def process(self, input: str) -> dict[str, Any]:
        """Answer an English query against the indexed corpus, or refuse below threshold."""
        self._require_ready()
        question = (input or "").strip()
        if not question:
            return {"answer": NO_ANSWER, "passages": [], "max_sim": 0.0,
                    "below_threshold": True}

        max_sim, hits = self._search(question)
        if max_sim < self.sim_threshold:
            log.info("max_sim=%.3f < %.2f → refusing", max_sim, self.sim_threshold)
            return {"answer": NO_ANSWER, "passages": hits, "max_sim": max_sim,
                    "below_threshold": True}

        context = "\n\n".join(
            f"[{i+1}] ({h['title']}) {h['text']}" for i, h in enumerate(hits)
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        out = self.llm(prompt, max_tokens=self.max_tokens, temperature=0.2,
                       stop=["</s>", "[INST]"])
        answer = out["choices"][0]["text"].strip()
        return {
            "answer": answer,
            "passages": hits,
            "max_sim": max_sim,
            "below_threshold": False,
            "question": question,
        }
