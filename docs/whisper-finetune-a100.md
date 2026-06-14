# Whisper fine-tune — A100 80 GB profile

The `Whisper.ipynb` notebook ships tuned for a large GPU (~80 GB VRAM, ≥64 GB RAM — A100, H100, or rented equivalent). This doc records the chosen knobs and the reasoning, so the next person reviewing the notebook doesn't have to reverse-engineer them.

For the free-tier T4 recipe, see `docs/colab-finetune.md`. For data-prep gotchas see `docs/whisper-data-prep-gotchas.md`. For how to evaluate the resulting checkpoint without getting fooled by raw WER see `docs/yoruba-wer-evaluation.md`.

## Final settings (as committed)

| Knob | Upstream / T4 default | A100 profile | Why |
|---|---|---|---|
| `target_modules` | `q_proj, v_proj` | `q_proj, k_proj, v_proj, out_proj, fc1, fc2` | Covers all attention + MLP projections so the adapter can actually shift Yorùbá phoneme/diacritic distributions, not just attend differently. Bumps trainable params to ~115 M (≈7% of the base). |
| `use_gradient_checkpointing` | `True` | `"unsloth"` | **Cannot be turned off** at batch 48 — see "The 78.6 GB OOM" below. |
| `per_device_train_batch_size` | 1 | 48 | Effective batch 48 with checkpointing fits comfortably on 80 GB. Without checkpointing this OOMs. |
| `gradient_accumulation_steps` | 4 | 1 | Real batch already 48 — no need to fake it. |
| `max_steps` | 60 | `-1` | Smoke-run cap removed. |
| `num_train_epochs` | (unset) | 3 | Hidi-agili `yoruba_tts_dataset` is ~9k clips in the current shard, ~561 total steps at batch 48. |
| `lr_scheduler_type` | linear | cosine | Smoother decay over a real schedule. |
| `warmup_steps` | 5 | `warmup_ratio=0.05` | Scales with run length instead of being a fixed 5 steps. |
| `optim` | `adamw_8bit` | `adamw_torch_fused` | 8-bit optimizer saves VRAM we don't need; fused AdamW is faster on A100+. |
| `bf16` / `fp16` | fp16 unless bf16 | bf16 (A100 supports it) + `tf32=True` | bf16 has wider dynamic range than fp16; tf32 speeds up matmuls. |
| `dataloader_num_workers` | 0 | `cpu_count()//2` | Default 0 starves the GPU during audio-feature loading. |
| `dataloader_pin_memory` | False | True | Faster host→device copies. |
| `dataloader_persistent_workers` | False | True | Workers don't get torn down each epoch. |
| Eval / save | every 5 steps, no best-model load | `eval_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="wer"` | Step-5 eval was 56 eval passes per epoch on the original config — hours of wasted decoding. Epoch-level eval matches the actual run length. |
| Data prep | per-example list comp, single-threaded | `dataset.map(batched=True, batch_size=32, num_proc=1)` | See `docs/whisper-data-prep-gotchas.md` for why this is the *correct* num_proc on Colab/cloud notebooks. |

## The 78.6 GB OOM — why gradient checkpointing must stay on

First attempt: batch 48 + checkpointing OFF on an 80 GB A100. Result:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 704.00 MiB.
GPU 0 has a total capacity of 79.25 GiB of which 646.81 MiB is free.
Including non-PyTorch memory, this process has 78.61 GiB memory in use.
```

The math:

- Model weights (bf16): ~3.1 GB
- AdamW state (fp32, 2× per trainable param): `115e6 × 2 × 4 ≈ 0.9 GB`
- Gradients (fp32 for trainable params): ~0.5 GB
- **Activations at batch 48, all-projections LoRA, no checkpointing: ~50–60 GB**

Activation memory scales as `batch × seq_len × hidden_dim × num_layers × (1 if checkpointed else many)`. Whisper-large-v3 is 32 enc + 32 dec layers, encoder seq is fixed at 1500 (80×3000 mel padded to 30 s), and we're now creating LoRA-A/B activations on q,k,v,out_proj,fc1,fc2 in every block. The cumulative footprint is what eats the card.

Unsloth's `"unsloth"` checkpointing variant drops activations to checkpoint boundaries only — roughly an order of magnitude smaller — at ~15–20% step-time cost. Same batch 48 fits comfortably in ~50 GB peak.

**Rule of thumb for LoRA on transformers**: whenever you change *any* of {batch, sequence length, num layers covered by LoRA}, re-check whether checkpointing should be on. Weights and optimizer are negligible; activations are where you OOM.

## Why the original 4-hour run never finished an epoch (on 80 GB)

The 80 GB card was irrelevant — the original settings couldn't *use* a big GPU:

1. **Batch 1 + grad accum 4** keeps an A100 at maybe 5% utilization. The model is the same size at batch 1 or batch 48 — only activations grow. At batch 1 you're paying for an A100 to run a T4 workload.
2. **Single-threaded data prep**. The original `[formatting_prompts_func(ex) for ex in dataset]` was one CPU core processing 9 k clips serially. With unknown dataset size, that alone can sit for an hour.
3. **`eval_steps=5` + `eval_strategy="steps"`**. On a ~280-step epoch that's 56 full passes over the eval set. Hours of decoding, not training.
4. **`logging_steps=1`** flushes every step — visually makes a fast run look slow.

The current notebook fixes all four.

## Expected wall-clock

Rough numbers on A100 80 GB, Hidi-agili (~9k clips, 6% test split):
- Data prep (`batched=True, num_proc=1`): ~30–60 s.
- 3 epochs at batch 48, checkpointing on: ~20–30 min total training.
- 3 evals at epoch boundaries: ~1–2 min total.

If GPU utilisation sits below 50% during training, the bottleneck is the dataloader — bump `dataloader_num_workers` further. If it's pinned at 100% but slow, you've hit the unavoidable checkpointing tax.

## Reverting to the T4 smoke run

If you re-open the notebook on a free Colab T4, only the trainer cell + the LoRA cell need to change back:

```python
# LoRA cell
target_modules = ["q_proj", "v_proj"]
use_gradient_checkpointing = "unsloth"   # already correct

# Trainer cell
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
max_steps = 60
# remove num_train_epochs, warmup_ratio
warmup_steps = 5
optim = "adamw_8bit"
lr_scheduler_type = "linear"
eval_strategy = "no"   # don't burn T4 minutes on eval during a smoke run
```

Everything else (bf16 auto-detect, batched data prep) is fine on T4 too — `is_bf16_supported()` flips to fp16 automatically.
