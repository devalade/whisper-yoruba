# Methodology

## Problem statement

A Yoruba speaker should be able to ask a question by voice and receive a spoken Yoruba answer grounded in an English-language knowledge base. The user is assumed to have limited or no English literacy; the system must therefore handle the full audio-to-audio interaction in Yoruba while drawing factual content from English sources.

This decomposes into five sub-problems:

1. **Speech recognition** — convert spoken Yoruba audio to text.
2. **Diacritic restoration** — recover tonal marks lost or omitted by ASR.
3. **Translation (YO→EN)** — translate the diacritized query into English for retrieval.
4. **Retrieval-augmented generation** — fetch relevant passages from English Wikipedia and synthesize an answer.
5. **Speech synthesis** — render the (translated back to Yoruba) answer as speech.

## System architecture

```
WAV (Yoruba, 16 kHz mono)
        │
        ▼
 ┌──────────────────┐    raw Yoruba text
 │ M1  ASR          │    (no/partial diacritics)
 │ Whisper-large-v3 │
 │  + Yoruba LoRA   │
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐    diacritized Yoruba
 │ M2  Diacritic    │
 │ Davlan mT5-base  │
 │  Yoruba ADR      │
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐    English query
 │ M3  YO→EN MT     │
 │ NLLB-200-600M    │
 └──────────────────┘
        │
        ▼
 ┌────────────────────────┐
 │ M4  RAG                │    English answer
 │  FAISS + MiniLM-L6     │
 │  Mistral-7B-Instruct   │
 │  English Wikipedia     │
 └────────────────────────┘
        │
        ▼
 ┌──────────────────┐    Yoruba answer audio
 │ M5  TTS chain    │
 │  NLLB EN→YO      │
 │  Davlan mT5 ADR  │
 │  MMS-TTS-yor     │
 └──────────────────┘
        │
        ▼
WAV (Yoruba)
```

## Component choices

### M1 — Speech recognition

**Chosen:** OpenAI `whisper-large-v3` fine-tuned on Yoruba with PEFT LoRA (r=32, alpha=64, target `q_proj` + `v_proj`).

**Alternatives considered:**
- `wav2vec2-xls-r-300m` Yoruba fine-tunes — competitive WER on read speech but no in-built language identification, smaller community of pretrained checkpoints, and weaker handling of long-form audio.
- `whisper-small` / `whisper-medium` — significantly faster to train (factor 6× and 2× respectively) but consistently 5–10 absolute WER points worse on FLEURS yo_ng in our pilot tests.
- Pre-existing `RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT` — used as a baseline for comparison, but trained on a different (and proprietary) corpus; our fine-tune lets us own the recipe.

**Why LoRA over full fine-tuning:**
- Tractable on a 32 GB consumer GPU (RTX 5090) at ~$2/run. Full fine-tuning of large-v3 requires 80 GB VRAM.
- Lower overfitting risk on a 9.5k-sample corpus — only ~1% of parameters are trainable, regularising the optimisation.
- Adapters are ~60 MB on disk, easy to version and share independently from the base.

**Training-target decision:** Train on diacritized Yoruba text, even though the original M1 contract was "non-diacritized text, M2 restores." Two reasons: (a) the chosen corpus comes with diacritics, stripping them would discard ground-truth signal; (b) M2's mT5 ADR is robust to already-diacritized input — it is effectively idempotent — so the chain stays correct end-to-end.

### M2 — Diacritic restoration

**Chosen:** `Davlan/mT5_base_yoruba_adr` — a public mT5-base fine-tuned for Yoruba automatic diacritic restoration.

**Why this model:** Davlan's checkpoint reports 64.63 BLEU on Global Voices and 70.27 BLEU on Menyo-20k. It is the best openly available Yoruba ADR model at time of writing, and the model card and accompanying paper provide a reproducible baseline.

**Why we keep M2 even after training M1 on diacritized targets:** defence-in-depth — when the fine-tuned M1 produces partial or noisy diacritics, M2 normalises them. The compute cost is negligible relative to M1.

### M3 — Yoruba → English translation

**Chosen:** `facebook/nllb-200-distilled-600M` with source language `yor_Latn`, target `eng_Latn`.

**Alternatives considered:**
- `Helsinki-NLP/opus-mt-mul-en` — older, smaller, lower BLEU on African languages per the NLLB paper.
- Larger NLLB checkpoints (1.3B, 3.3B) — better quality but disproportionate cost for our use case (single-query throughput, M3 is not the bottleneck).

**Why NLLB:** broad African language coverage, single model handles 200 languages so the same checkpoint is reused in M5 (EN→YO leg), and the 600M distilled variant runs comfortably on Apple Silicon.

### M4 — Retrieval-augmented generation

**Chosen pipeline:** FAISS `IndexFlatIP` + `sentence-transformers/all-MiniLM-L6-v2` embeddings (384-dim) for retrieval over chunked English Wikipedia passages; `Mistral-7B-Instruct-v0.2` (Q4_K_M GGUF via `llama.cpp`) for answer synthesis.

**Knowledge base:** Currently a 12-article English Wikipedia seed corpus (~2k chunks of 200 tokens, 50-token overlap). Production scope would expand this to a larger snapshot.

**Why this stack:**
- MiniLM-L6 — high quality at small size (84 MB), runs fast on CPU/MPS, well-studied baseline.
- FAISS IndexFlatIP — exact cosine similarity (after L2 normalisation), no recall loss from approximate search at this corpus size.
- Mistral-7B Q4 — local inference, no API cost, fits in ~5 GB RAM, strong instruction following.

**Why cross-lingual retrieval rather than a Yoruba-only knowledge base:** the Yoruba Wikipedia contains roughly 30k articles versus the English Wikipedia's 6.7M, a 200× difference in coverage. Translating the user query to English at the boundary expands the answerable question space by two orders of magnitude.

### M5 — Speech synthesis

**Chosen chain:** NLLB EN→YO translation → Davlan mT5 ADR (re-applied to ensure diacritics on the model output) → `facebook/mms-tts-yor` (a VITS-based Yoruba TTS model).

**Why MMS-TTS-yor:** the only openly available high-quality Yoruba TTS model at time of writing, part of Meta's Massively Multilingual Speech release covering 1100+ languages.

**Known limitation:** single speaker, single style. Output audio quality is intelligible but not natural — this is acknowledged as a system-level limitation tied to the state of open Yoruba TTS, not something a thesis-scope intervention can solve.

## Data

### Fine-tuning corpus (M1)

`Hidi-agili/yoruba_tts_dataset` on HuggingFace:
- 9,499 (audio, text) pairs
- Single speaker, 16 kHz mono, durations 0.3–23.3 s
- Yoruba text with full diacritics preserved
- Open license

**Why this dataset:** cleanest open Yoruba speech corpus with reliable diacritized transcriptions. Trade-off: single speaker limits acoustic diversity (no robustness to accent, noise, or speaker variation).

### Evaluation corpus

`google/fleurs` (yo_ng subset) — FLEURS is the de facto multilingual ASR benchmark, validation split used. Yoruba evaluation references are diacritized; WER is computed after stripping diacritics from both reference and hypothesis to compare against the existing baseline fairly (the diacritic restoration is then a separate M2 metric).

### Knowledge base (M4)

12 English Wikipedia articles, manually chosen for thematic coverage, chunked at 200 tokens with 50-token overlap. Production expansion is straightforward (`scripts/build_index.py` is corpus-agnostic).

## Training setup

| | |
|---|---|
| Base model | `openai/whisper-large-v3` (1.55B params) |
| Adapter | PEFT LoRA, r=32, α=64, target `q_proj`+`v_proj`, dropout 0.05 |
| Trainable params | ~15M (~1% of base) |
| Optimiser | AdamW, lr 1e-5, warmup 50 steps |
| Precision | bfloat16 |
| Gradient checkpointing | Yes, `use_reentrant=False` (required for frozen base + adapter) |
| Batch size | 12 (per device), grad accum 2 → effective 24 |
| Epochs | 3 |
| Hardware | 1× RTX 5090 (32 GB VRAM) on Vast.ai |
| Wall time | ~2.5 hours |
| Cost | ~$2 |

## Evaluation methodology

1. **M1 ASR quality:** Word Error Rate on FLEURS yo_ng validation, n=200 samples. Computed via `jiwer` after normalising both reference and hypothesis with diacritic stripping, lowercasing, and punctuation removal. Compared against (a) base `whisper-large-v3` and (b) `RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT`.

2. **End-to-end qualitative evaluation:** 10 hand-curated Yoruba audio queries are run through the full pipeline. Each example is recorded with: ASR transcript, diacritized form, English translation, retrieved passages, generated English answer, translated Yoruba answer, synthesized audio. A native Yoruba speaker (the thesis author) judges each on a 1–5 fluency/correctness scale.

3. **Latency profile:** wall-clock time per stage on the author's Apple Silicon machine, reported as median of 5 runs per stage.

(Note: a formal user study and crowd-sourced acoustic robustness evaluation are out of scope for this thesis under the one-week timeline. These are explicitly listed in "Future work.")
