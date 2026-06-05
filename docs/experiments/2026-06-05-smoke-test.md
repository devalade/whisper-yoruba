# Experiment: pipeline-validation smoke test

**Date:** 2026-06-05
**Goal:** Prove the full LoRA fine-tuning pipeline runs end-to-end on a Blackwell (RTX 5090) Vast.ai instance.
**Outcome:** ✅ Pipeline works. Numbers are meaningless (50 samples / 1 epoch); the value of the run was catching environment bugs.

## Setup

| | |
|---|---|
| Base model | `openai/whisper-large-v3` |
| Adapter | LoRA r=32, α=64, target q_proj+v_proj, dropout 0.05 |
| Dataset | `Hidi-agili/yoruba_tts_dataset`, 50 train / 10 eval samples |
| Epochs | 1 |
| Batch size | 8 per device, grad accum 2 |
| Learning rate | 1e-5 |
| Precision | bf16 |
| Gradient checkpointing | yes (use_reentrant=False) |
| GPU | 1× RTX 5090 32 GB on Vast.ai |
| Cost | ~$0.20 (debugging took longer than the run itself) |
| Wall time | ~15 seconds training + setup time |

## Results

```
{'eval_loss': 5.153919219970703,
 'eval_wer': 0.8835616438356164,
 'eval_runtime': 4.369,
 'eval_samples_per_second': 2.289,
 'epoch': 1.0}

{'train_runtime': 15.4199,
 'train_samples_per_second': 3.243,
 'train_loss': 4.175473213195801,
 'epoch': 1.0}
```

WER 88% is expected on 50 samples / 1 epoch — meaningless as quality signal. The pipeline ran cleanly: model loaded, dataset preprocessed, training loop completed, eval ran, adapter saved.

## What worked

- The full stack runs on a fresh Vast.ai box once the install dance is correct.
- bf16 on Blackwell works as expected with non-reentrant gradient checkpointing.
- The HF datasets + DataCollatorSpeechSeq2SeqWithPadding + Trainer combination is correct.

## What didn't / what we fixed along the way

Five environment bugs surfaced and were fixed in the repo:

1. `mlx-whisper` installing on Linux and breaking `transformers`' tensor-type sniffing → platform-markered as macOS-only.
2. `datasets` 4.x requiring `torchcodec` (no wheel for `torch 2.12+cu130`) → pinned `datasets<4`.
3. PEFT `LoraConfig(task_type="SEQ_2_SEQ_LM")` injecting `input_ids` into Whisper → dropped `task_type`.
4. Reentrant gradient checkpointing requiring inputs with `requires_grad`, incompatible with PEFT-frozen base → switched to `use_reentrant=False`.
5. Vast image setting `HF_HOME=/workspace/.hf_home`, not `~/.cache/huggingface` → cleanup commands now target the right path.

All five documented in `docs/troubleshooting.md`.

Additionally, the `Makefile`'s default `PYTHON` path is a local Mac conda path — overridden on the box with `export PYTHON=python`. Documented in `docs/vast-ai-setup.md`.

## Decision

✅ Pipeline validated. Next run (3 epochs, full dataset) is the one that will produce a checkpoint worth shipping. Tracking as `2026-06-06-large-v3-yoruba-ft.md`.
