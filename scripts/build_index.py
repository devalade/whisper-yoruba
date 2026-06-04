"""Fetch English Wikipedia articles relevant to the Yoruba domain, chunk them
into 200-token windows (50-token overlap), embed with MiniLM-L6-v2, and write
a FAISS IndexFlatIP + a passages.jsonl side-file consumed by M4.

Run: python -m scripts.build_index
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

import config

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_EXTRACT_API = "https://en.wikipedia.org/w/api.php"
INDEX_PATH = config.WIKI_DIR / "faiss.index"
META_PATH = config.WIKI_DIR / "passages.jsonl"

# Seed corpus: topics a Yoruba speaker might naturally ask about.
ARTICLES = [
    "Yoruba people",
    "Yoruba language",
    "Yoruba religion",
    "Lagos",
    "Nigeria",
    "Ile-Ife",
    "Oyo Empire",
    "Wole Soyinka",
    "Fela Kuti",
    "Olumo Rock",
    "Olusegun Obasanjo",
    "Chinua Achebe",
]


def fetch_article(title: str) -> str:
    """Pull the full plain-text extract for a Wikipedia article."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
        "redirects": 1,
    }
    r = requests.get(
        WIKI_EXTRACT_API,
        params=params,
        headers={"User-Agent": "yoruba-pipeline/0.1 (research)"},
        timeout=30,
    )
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""


def chunk_by_tokens(text: str, tokenizer, max_tokens: int, overlap: int) -> list[str]:
    """Token-level sliding window using the embedder's own tokenizer for
    consistent token counts at retrieval time."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = max_tokens - overlap
    chunks = []
    for start in range(0, len(ids), step):
        window = ids[start : start + max_tokens]
        if not window:
            break
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + max_tokens >= len(ids):
            break
    return chunks


def main() -> None:
    config.WIKI_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading embedder {config.M4_EMBED_MODEL}")
    embedder = SentenceTransformer(config.M4_EMBED_MODEL)
    tokenizer = embedder.tokenizer

    passages: list[dict] = []
    for title in ARTICLES:
        print(f"fetching: {title}")
        text = fetch_article(title)
        if not text:
            print(f"  empty extract for {title}, skipping")
            continue
        chunks = chunk_by_tokens(
            text, tokenizer,
            max_tokens=config.M4_CHUNK_TOKENS,
            overlap=config.M4_CHUNK_OVERLAP,
        )
        for i, chunk in enumerate(chunks):
            passages.append({"title": title, "chunk_id": i, "text": chunk})
        print(f"  {len(chunks)} chunks")

    print(f"\nembedding {len(passages)} passages…")
    texts = [p["text"] for p in passages]
    vecs = embedder.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True, batch_size=64,
    ).astype("float32")

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, str(INDEX_PATH))
    with META_PATH.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nwrote {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
    print(f"wrote {META_PATH}")


if __name__ == "__main__":
    main()
