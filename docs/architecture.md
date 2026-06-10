# Architecture — how the modules fit together

A map of the runtime. Read this before touching `pipeline.py` or a module —
it shows what each stage owns, what data crosses each boundary, and where the
fine-tuned Whisper model plugs in.

For *why* these choices were made, see `methodology.md`. For *how to train the
M1 model*, see `fine-tuning-runbook.md`. This document is purely about
mechanics.

## One-paragraph summary

The pipeline is five modules chained `M1 → M2 → M3 → M4 → M5`. Each is a class
that obeys the same `PipelineModule` contract (`initialize()` once,
`process()` per turn). `pipeline.py` owns one instance of each, wires the
output of one stage into the input of the next, and writes a JSONL log entry
per stage. Two stages have swappable backends (M1: mlx vs HF; M4: RAG vs
chat) selected by CLI flag; everything else is fixed.

## The module contract

Every module in `modules/` inherits from `PipelineModule` (`modules/base.py`):

| Method            | Called      | Purpose                                                    |
| ----------------- | ----------- | ---------------------------------------------------------- |
| `initialize()`    | once        | Load weights, allocate device buffers, set `_ready = True` |
| `process(input)`  | per turn    | Run inference on a single input, return a dict             |
| `_require_ready()`| internal    | Guard that yells if you call `process` before init         |

This contract is the only thing the pipeline assumes. As long as a new
implementation honours it and returns the same dict shape as the one it
replaces, it drops in without changes elsewhere. That's how `M1ASR` and
`M1ASRHF` coexist, and how `M4RAG` and `M4Chat` are interchangeable.

## End-to-end data flow

```
                    ┌─────────────────────────────────────────────────────┐
                    │                pipeline.py / YorubaPipeline         │
                    └─────────────────────────────────────────────────────┘

  input.wav  ──►  M1  ──►  raw YO text  ──►  M2  ──►  diacritized YO text
                  ASR                         diacritic
                                                            │
                                                            ▼
                                                    M3  ──►  EN query
                                                    YO→EN
                                                            │
                                                            ▼
                                                    M4  ──►  EN answer
                                                    RAG or chat
                                                            │
                                                            ▼
                                              M5 (EN→YO  ──►  M2 again
                                              + diacritic ──►  + MMS-TTS)
                                                            │
                                                            ▼
                                                     output.wav
```

Notes on the diagram:

- **M2 runs twice.** Once between M1 and M3 to clean up Whisper's
  no-diacritics output before translation, and once inside M5 to clean up
  NLLB's EN→YO output before the TTS model. Same class, one instance,
  invoked at two points. The second call is wrapped inside `M5TTS`.
- **The boundary between M3 and M4 is English text.** This is the seam that
  lets us reuse English-language RAG, LLMs, and knowledge bases against a
  Yoruba user-facing surface. It is the central architectural bet of the
  project.
- **All `process` outputs are dicts**, not bare strings — even when only one
  field is consumed downstream. This is so the per-stage JSONL log can
  capture timings, confidences, retrieval scores, etc., without changing the
  call signature.

## Stage-by-stage reference

### M1 — ASR (`modules/m1_asr.py`, `modules/m1_asr_hf.py`)

| | |
| - | - |
| In  | path to a 16 kHz mono WAV |
| Out | `{"text": str, "segments": [...], "language": str, ...}` |
| Models | `mlx-community/whisper-large-v3-mlx` (mlx, default) or `devalade/whisper-large-v3-yoruba` (HF, fine-tuned) |
| Device | MPS via mlx-whisper, or auto-selected (CUDA / MPS / CPU) via transformers |

Two backends, same output shape:

- **`M1ASR` (mlx-whisper)** — fastest on Apple Silicon, but uses the stock
  multilingual checkpoint. Good baseline.
- **`M1ASRHF` (transformers)** — loads our own Yoruba fine-tune. The
  fine-tuning script (`scripts/finetune_whisper.py`) is what produces the
  checkpoint that gets pushed to HF Hub and named here. See
  `fine-tuning-runbook.md`.

Pick via `pipeline.py --asr mlx|hf`. Default is `mlx`.

The text M1 returns is **not** diacritized — Whisper's output omits Yoruba
tone and sub-dot marks. That's M2's job.

### M2 — Diacritic restoration (`modules/m2_diacritic.py`)

| | |
| - | - |
| In  | bare Yoruba text (no diacritics) |
| Out | `{"text": str}` — same text with tone marks and sub-dots restored |
| Model | `Orifeoluwafemi/yoruba-text-diacritization` (HF seq2seq) |
| Device | MPS / CUDA / CPU, auto-selected |

Used at two points (see flow diagram). One instance is created in
`YorubaPipeline.__init__`; `M5TTS` constructs its own instance internally for
the second pass.

### M3 — Translation YO → EN (`modules/m3_translate.py`)

| | |
| - | - |
| In  | diacritized Yoruba text |
| Out | `{"text": str}` — English translation |
| Model | NLLB-200 distilled 600M |
| Device | MPS / CUDA / CPU, auto-selected |

A thin wrapper around `AutoModelForSeq2SeqLM` with NLLB language codes set
in config (`M3_SRC_LANG`, `M3_TGT_LANG`).

### M4 — Answer generation (`modules/m4_rag.py`, `modules/m4_chat.py`)

Two interchangeable implementations, same output shape:

**`M4RAG`** (default):

| | |
| - | - |
| In  | English question |
| Out | `{"answer": str, "max_sim": float, "hits": [...]}` |
| Models | MiniLM-L6-v2 (embeddings) + Mistral-7B-Instruct Q4 via llama.cpp |
| Index | FAISS `IndexFlatIP` over Wikipedia passages (`data/wikipedia/`) |

Embeds the query, retrieves top-K passages, builds the `[INST] … [/INST]`
prompt with retrieved context, calls llama.cpp. If `max_sim` is below
`config.M4_SIM_THRESHOLD` the model is instructed to answer
`"I don't know based on the provided context."` rather than hallucinate.

**`M4Chat`** (selected with `--chat`):

Same output shape, but no retrieval — prompts Mistral directly. Used for
free-form chat and as an A/B baseline against RAG.

The pipeline chooses `M4RAG` or `M4Chat` once at construction time
(`YorubaPipeline(chat_mode=...)`); they are never both loaded.

### M5 — TTS leg (`modules/m5_tts.py`)

| | |
| - | - |
| In  | English answer (string) + `output_path` (WAV) |
| Out | `{"audio_path": Path, "yo_diacritized": str, "duration_s": float, ...}` |
| Models | NLLB EN→YO + M2 diacritic + Facebook MMS-TTS Yoruba |
| Device | MPS / CUDA / CPU, auto-selected |

M5 is itself a small pipeline: it translates EN→YO with NLLB, runs the
result through an internal `M2Diacritic` instance, then synthesises the WAV
with MMS-TTS. It's the only stage that writes a file — everything else is
in-memory.

## How `pipeline.py` wires it all

`YorubaPipeline` (`pipeline.py:26`) is ~50 lines and does three jobs:

1. **`__init__`** — instantiate one of each module, picking the M1 backend
   and M4 mode from constructor flags. No I/O yet.
2. **`initialize()`** — call each module's `initialize()` in M1→M5 order.
   This is the slow path (model weights). Done once, before the first turn.
3. **`run(audio_in, audio_out)`** — execute one turn. Threads dict outputs
   into the next stage's input and calls `log_stage(...)` after every step.
   Returns the full per-stage dict for callers that want it (mic loop,
   tests, scripts).

The mic loop (`_run_mic_loop`) wraps `run()` in a record / process / play
cycle and handles SIGINT cleanly around PortAudio.

## Where the fine-tune fits

The Whisper fine-tune is **not** part of the runtime pipeline — it's an
offline process that produces a checkpoint name, which is then dropped into
`config.M1_HF_MODEL`. Sequence:

1. `scripts/finetune_whisper.py` runs on a CUDA box (Vast.ai 5090).
2. Output adapter goes to `config.FT_OUTPUT_DIR`.
3. `scripts/merge_lora.py` merges the LoRA into a plain Whisper checkpoint.
4. The merged checkpoint is pushed to HF Hub.
5. The Hub repo id is set as `config.M1_HF_MODEL`.
6. At runtime, `pipeline.py --asr hf` selects `M1ASRHF`, which loads that
   model.

Everything downstream of M1 is unaffected by which ASR backend is in use,
because both backends honour the same `{"text": ...}` output contract.

## Per-stage logging

`log_stage(stage, run_id, payload)` (from `utils.logging`) appends a single
JSONL line per stage per turn to `logs/<run_id>.jsonl`. The payload is the
stage's full output dict, so a single log file reconstructs the entire
turn — input text, intermediate translations, retrieval hits, generated
WAV path, durations. Used for eval, debugging, and the thesis appendix.

## Variant matrix

What you can flip without touching code:

| Flag                | Effect                                       | Where it lands              |
| ------------------- | -------------------------------------------- | --------------------------- |
| `--asr mlx`         | mlx-whisper large-v3 (default)               | `M1ASR`                     |
| `--asr hf`          | HF Yoruba fine-tune                          | `M1ASRHF`                   |
| `--chat`            | Skip retrieval, prompt Mistral directly      | `M4Chat` replaces `M4RAG`   |
| `--mic`             | Push-to-talk loop                            | `_run_mic_loop`             |
| (positional WAV)    | Single-shot file-in / file-out               | `YorubaPipeline.run`        |

Anything else (changing models, beam widths, prompts, FAISS top-K) requires
editing `config.py` — there is no per-module CLI surface beyond the four
flags above.

## Related docs

- `methodology.md` — *why* this five-stage decomposition.
- `fine-tuning-runbook.md` — how to produce the M1 HF checkpoint.
- `small-run-with-holdout.md` — fast iteration recipe for M1 fine-tunes.
- `conversational-eval.md` — eval set for M1 beyond FLEURS.
- `troubleshooting.md` — known bugs and their fixes.
- `vast-ai-setup.md` — the GPU box that runs the fine-tune.
