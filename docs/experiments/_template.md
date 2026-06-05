# Experiment: <short descriptive name>

**Date:** YYYY-MM-DD
**Goal:** One sentence on what this run is trying to learn or achieve.
**Outcome:** ✅ Shipped to HF / ⚠️ Useful but didn't ship / ❌ Failed — see below

## Setup

| | |
|---|---|
| Base model | `openai/whisper-large-v3` |
| Adapter | LoRA r=32, α=64, target q_proj+v_proj, dropout 0.05 |
| Dataset(s) | `Hidi-agili/yoruba_tts_dataset` (9,499 rows) |
| Epochs | 3 |
| Batch size | 12 per device, grad accum 2 |
| Learning rate | 1e-5 (warmup 50 steps) |
| Precision | bf16 |
| Gradient checkpointing | yes (use_reentrant=False) |
| GPU | 1× RTX 5090 32 GB on Vast.ai |
| Cost | $X.XX |
| Wall time | Xh Xm |

## Results

### Training curve

| Epoch | Train loss | Eval WER |
|---|---|---|
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |

### Held-out eval (FLEURS yo_ng validation, n=200)

| Model | WER |
|---|---|
| `openai/whisper-large-v3` (zero-shot baseline) |  |
| `RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT` (prior baseline) |  |
| **This run (devalade/whisper-large-v3-yoruba)** |  |

## Qualitative samples (3–5)

| Reference (yo) | Hypothesis (yo) | Notes |
|---|---|---|
|  |  |  |
|  |  |  |

## What worked

- 

## What didn't / what would I change

- 

## Decision

✅ Pushed to HF as `devalade/whisper-large-v3-yoruba`. Updated `config.M1_HF_MODEL`.

(Or, if it didn't ship: ❌ Worse than baseline by N points — kept baseline. Reasons: ...)
