# Experiment: first full fine-tune of whisper-large-v3 on Yoruba

**Date:** 2026-06-05
**Goal:** Produce the first thesis-quality Yoruba fine-tune of `openai/whisper-large-v3` and replace the prior baseline (`RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT`) as M1's HF backend.
**Outcome:** ✅ Shipped to HF as [`devalade/whisper-large-v3-yoruba`](https://huggingface.co/devalade/whisper-large-v3-yoruba). `config.M1_HF_MODEL` switched in commit `c33a951`.

## Setup

| | |
|---|---|
| Base model | `openai/whisper-large-v3` |
| Adapter | LoRA r=32, α=64, target q_proj+v_proj, dropout 0.05 |
| Dataset | `Hidi-agili/yoruba_tts_dataset` (9,499 rows, single speaker, TTS-read) |
| Epochs | 3 |
| Batch size | 12 per device, grad accum 2 |
| Learning rate | 1e-5 (warmup 50 steps) |
| Precision | bf16 |
| Gradient checkpointing | yes (use_reentrant=False) |
| GPU | 1× RTX 5090 32 GB on Vast.ai |
| Cost | _TBD — fill from Vast billing_ |
| Wall time | ~2–2.5 h expected per runbook — _TBD: confirm actual_ |

## Results

### Training curve

| Epoch | Train loss | Eval WER |
|---|---|---|
| 1 | _TBD_ | _TBD_ |
| 2 | _TBD_ | _TBD_ |
| 3 | _TBD_ | _TBD_ |

> Reconstruct from `models/whisper-yo-lora/trainer_state.json` on the Vast box, or from the TensorBoard logs if they were copied back. If neither survives, mark unrecoverable and rely on the held-out eval below.

### Held-out eval (FLEURS yo_ng validation, n=200)

| Model | WER |
|---|---|
| `openai/whisper-large-v3` (zero-shot baseline) | _TBD — run `make wer-hf N=200` with base model_ |
| `RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT` (prior baseline) | _TBD_ |
| **This run (`devalade/whisper-large-v3-yoruba`)** | _TBD — run `make wer-hf N=200`_ |

## Qualitative samples

The conversational regression that motivated the follow-up tooling (`2026-06-06-small-run-holdout.md`):

| Reference (yo) | Hypothesis (yo) | Notes |
|---|---|---|
| `Kíni orúkọ rẹ?` | `Kini Oru gọ ọrẹ?` | Word-boundary failure on short conversational input. Distribution mismatch: training corpus is studio-read, this is microphone input. |

## What worked

- Full LoRA stack ran cleanly on Blackwell after the five environment fixes from the `2026-06-05-smoke-test` run.
- Adapter merged and pushed to HF without issue; pipeline picked it up by changing `config.M1_HF_MODEL` only.

## What didn't / what would I change

- FLEURS yo_ng (studio-read news prose) hides the distribution that actually matters for the pipeline — short, microphone-recorded conversational queries. The first end-to-end test surfaced this; **this is why `docs/conversational-eval.md` exists.**
- Single-speaker training corpus is the deepest constraint here. No recipe tweak on this dataset will fix acoustic diversity.

## Decision

✅ Pushed to HF as `devalade/whisper-large-v3-yoruba`. `config.M1_HF_MODEL` updated. Baseline preserved as a comment in `config.py` for quick A/B.

Follow-up: build the small-run + conversational-eval iteration loop before the next full run — tracked in `2026-06-06-small-run-holdout.md`.
