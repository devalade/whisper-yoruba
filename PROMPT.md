You are helping me build a multilingual voice query system for Yoruba speakers.
The system is a sequential pipeline with 5 modules:

M1 - ASR: Whisper Large v3 via mlx-whisper (Apple Silicon optimized)
M2 - Diacritic restoration: Orife seq2seq model (HuggingFace)
M3 - Translation YO→EN: NLLB-200 distill-600M (HuggingFace, yor_Latn → eng_Latn)
M4 - RAG: FAISS (IndexFlatIP) + MiniLM-L6-v2 embeddings + Mistral 7B Q4 (llama.cpp)
M5 - TTS: MMS-TTS yoruba (HuggingFace), preceded by EN→YO translation + M2 diacritization

Deployment environment:
- Apple M4 Pro, 24GB unified RAM
- macOS, Miniforge (ARM64), Python 3.11
- PyTorch with MPS backend
- All models run locally, no cloud dependency after setup

Pipeline flow:
Audio WAV (16kHz mono) → M1 → raw Yoruba text → M2 → diacritized Yoruba
→ M3 → English query → M4 → English answer → M5 (EN→YO + M2 + TTS) → Yoruba audio WAV

Key architectural decisions:
- M2 is applied twice: once after M1 (before translation), once inside M5 (before TTS)
- FAISS similarity threshold: 0.5 (below = no answer rather than hallucination)
- Wikipedia EN passages: 200 tokens, 50-token overlap
- All intermediate results must be logged for error propagation analysis

Each module must be a class with two methods:
- initialize(): loads the model into memory
- process(input): runs inference and returns output

Start by scaffolding the project structure, then implement one module at a time,
starting with M1. Ask me before moving to the next module.
