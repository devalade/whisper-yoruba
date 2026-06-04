# Yoruba Voice Query Pipeline

End-to-end voice assistant for Yoruba speakers. Speak a question in Yoruba, get
a spoken Yoruba answer grounded in an English Wikipedia corpus. Runs fully
locally on Apple Silicon — no cloud calls after the initial model downloads.

```
WAV (yo, 16 kHz mono)
        │
        ▼
 ┌──────────────┐   raw Yoruba text
 │ M1  ASR      │ ─ Whisper Large v3 (mlx-whisper)
 └──────────────┘
        │
        ▼
 ┌──────────────┐   diacritized Yoruba
 │ M2  ADR      │ ─ Davlan/mT5_base_yoruba_adr
 └──────────────┘
        │
        ▼
 ┌──────────────┐   English query
 │ M3  YO→EN    │ ─ NLLB-200 distilled-600M
 └──────────────┘
        │
        ▼
 ┌──────────────┐   English answer
 │ M4  RAG      │ ─ MiniLM-L6 + FAISS + Mistral-7B Q4 (llama.cpp)
 └──────────────┘
        │
        ▼
 ┌──────────────┐   Yoruba answer audio
 │ M5  TTS      │ ─ NLLB EN→YO  →  M2 diacritize  →  MMS-TTS-yor
 └──────────────┘
        │
        ▼
   WAV (yo)
```

M2 runs twice on purpose: once after ASR to clean the input for translation,
and once inside M5 to clean the NLLB EN→YO output before TTS.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4). Tested on M4 Pro / 24 GB.
- macOS with Miniforge (ARM64), Python 3.11.
- ~15 GB free disk for model weights.
- First run downloads several gigabytes from Hugging Face.

## Setup

```bash
# 1. Create env
conda create -n yoruba python=3.11 -y
conda activate yoruba

# 2. Install deps
make install               # or: pip install -r requirements.txt

# 3. Drop the Mistral GGUF into models/
#    Expected file: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
#    Source: TheBloke/Mistral-7B-Instruct-v0.2-GGUF on Hugging Face
mkdir -p models
# (download mistral-7b-instruct-v0.2.Q4_K_M.gguf into models/)

# 4. Build the Wikipedia FAISS index used by M4
make index                 # or: python -m scripts.build_index
# writes data/wikipedia/faiss.index and data/wikipedia/passages.jsonl
```

The seed corpus (in `scripts/build_index.py`) covers Yoruba people, language,
religion, Lagos, Nigeria, Ile-Ife, Oyo Empire, Wole Soyinka, Fela Kuti, Olumo
Rock, Olusegun Obasanjo, Chinua Achebe. Edit `ARTICLES` to add more topics, then
re-run the script.

## Run the pipeline

```bash
make run                                    # uses the FLEURS sample
make run INPUT=path/to.wav OUTPUT=out.wav   # custom files

# equivalent without make:
python pipeline.py <input.wav> [output.wav]
```

`<input.wav>` must be 16 kHz mono. `[output.wav]` defaults to
`data/outputs/response.wav`.

Grab a sample Yoruba clip from FLEURS first if you don't have one:

```bash
make sample        # or: python tests/fetch_yoruba_sample.py
make run
```

### Talk to it through your microphone

```bash
make talk
# or: python pipeline.py --mic
```

Conversation mode. Models load once, then the system waits for you on each
turn:

1. Press Enter to start speaking.
2. Press Enter again to stop.
3. The pipeline runs and the Yoruba answer auto-plays.
4. The prompt comes back for the next turn.

Type `q` + Enter at the prompt to quit (Ctrl+C also works). macOS will ask for
microphone permission the first time. Each turn's audio is saved as
`data/outputs/response_NNN.wav` and logged to `logs/run-mic_capture.jsonl`.

### Benchmark ASR backends (WER)

```bash
make wer-mlx N=50          # eval mlx-whisper Large v3 on 50 FLEURS yo_ng samples
make wer-hf  N=50          # eval the HF Yoruba fine-tune on the same set
make wer N=100             # run both back-to-back
```

Reference comes from `google/fleurs` (`yo_ng` config), split defaults to
`validation`. Both reference and hypothesis are normalized (lowercase,
diacritics stripped, punctuation removed) before scoring — fair to M1 since it
emits non-diacritized text and M2 handles diacritics later. Per-sample results
land in `logs/wer_<backend>.jsonl` with the aggregate WER on the first line.

### Swap the ASR backend (M1)

By default M1 uses `mlx-whisper` (Whisper Large v3, Apple-Silicon-optimized).
An alternate HuggingFace backend loads a Yoruba-fine-tuned Whisper Large v2
(`RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT`):

```bash
make talk-hf                      # RAG + HF Whisper
make chat-hf                      # free-form + HF Whisper
# or: python pipeline.py --mic --asr hf
# also on a file: python pipeline.py input.wav --asr hf
```

The HF backend auto-picks the right device/attention for the host: CUDA + fp16
+ flash-attn-2 when available, otherwise MPS or CPU + sdpa + fp32 (fp16 on MPS
is avoided — Whisper can produce NaNs there).

### Chat mode (no retrieval)

To bypass M4's Wikipedia RAG and let Mistral answer from its own knowledge:

```bash
make chat
# or: python pipeline.py --mic --chat
# also works on a file: python pipeline.py input.wav --chat
```

Same pipeline (M1→M2→M3→M4→M5), but M4 becomes `M4Chat` which prompts the local
Mistral directly with no retrieved context. Useful for open-ended questions
outside the indexed corpus. Trade-off: answers can be less factual and aren't
grounded in citable passages.

Each stage's intermediate result is appended to `logs/<run_id>.jsonl` for
error-propagation analysis (which stage degraded the output).

Expected console output:

```
=== M1 raw YO    ===   <whisper transcript>
=== M2 diacrit.  ===   <restored tone marks>
=== M3 EN query  ===   <English question>
=== M4 EN answer (max_sim=0.612) ===   <retrieved/generated English answer>
=== M5 YO answer ===   <Yoruba answer text>
WAV: data/outputs/response.wav  (4.81s)
log: logs/run-fleurs_yo_sample.jsonl
```

## Per-module testing

Each module can be exercised in isolation:

```bash
make test           # all per-module tests (M1..M5)
make test-m1        # individual module (test-m1 .. test-m5)
make test-chain     # M1→M3 and M1→M4 chained tests

# equivalent without make:
python -m tests.test_m1     # ASR only
python -m tests.test_m2     # diacritic restoration
python -m tests.test_m3     # YO→EN translation
python -m tests.test_m4     # RAG (requires the FAISS index)
python -m tests.test_m5     # EN answer → YO audio
```

## Project layout

```
config.py              model IDs, paths, thresholds
pipeline.py            YorubaPipeline class + CLI entrypoint
modules/
  base.py              shared Module interface
  m1_asr.py            Whisper Large v3 (mlx)
  m2_diacritic.py      mT5 Yoruba ADR
  m3_translate.py      NLLB YO→EN
  m4_rag.py            MiniLM + FAISS + Mistral-7B
  m5_tts.py            NLLB EN→YO + M2 + MMS-TTS
scripts/build_index.py Wikipedia fetch / chunk / embed / index
tests/                 per-module + chained sanity tests
utils/logging.py       JSONL stage logger
data/                  audio/, wikipedia/, outputs/
logs/                  per-run JSONL stage logs
models/                local GGUF weights
```

## Key knobs (`config.py`)

| Setting              | Default | Notes                                            |
| -------------------- | ------- | ------------------------------------------------ |
| `M4_SIM_THRESHOLD`   | 0.5     | Below this max cosine sim, M4 refuses to answer rather than hallucinate. |
| `M4_CHUNK_TOKENS`    | 200     | Passage window size (embedder's own tokenizer).  |
| `M4_CHUNK_OVERLAP`   | 50      | Sliding-window overlap.                          |
| `M4_TOP_K`           | 4       | Passages fed to Mistral as context.              |
| `M1_LANGUAGE`        | `yo`    | Forces Whisper into Yoruba mode.                 |

## Troubleshooting

- **`transformers` tokenizer error on mT5 or NLLB** — ensure `transformers<5`
  and that `sentencepiece` + `protobuf` are installed. Pinned in
  `requirements.txt`.
- **`faiss.index` missing** — run `python -m scripts.build_index`.
- **Mistral load fails** — confirm the GGUF path matches `config.M4_LLM_PATH`.
- **Whisper transcript is English** — input WAV may not actually be Yoruba, or
  not 16 kHz mono. Resample with `ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`.
- **M4 returns "no answer"** — `max_sim` is below `M4_SIM_THRESHOLD`. Either
  expand the Wikipedia seed corpus in `scripts/build_index.py` or lower the
  threshold.
