# Whisper fine-tune — A100 80 GB profile

The `Whisper.ipynb` notebook ships tuned for a large GPU (~80 GB VRAM, ≥64 GB RAM — A100, H100, or rented equivalent). This doc records the chosen knobs and the reasoning, so the next person reviewing the notebook doesn't have to reverse-engineer them.

For the free-tier T4 recipe, see `docs/colab-finetune.md`.

## What changed vs. the upstream Unsloth template

| Knob | Upstream / T4 default | A100 profile | Why |
|---|---|---|---|
| `target_modules` | `q_proj, v_proj` | `q_proj, k_proj, v_proj, out_proj, fc1, fc2` | Covers all attention + MLP projections so the adapter can actually shift Yorùbá phoneme/diacritic distributions, not just attend differently. |
| `per_device_train_batch_size` | 1 | 32 | whisper-large-v3 with rank-64 LoRA peaks around ~35 GB at batch 32 on 80 GB cards. |
| `gradient_accumulation_steps` | 4 | 1 | Effective batch already 32 — no need to fake it. |
| `max_steps` | 60 | `-1` | Smoke run cap removed. |
| `num_train_epochs` | (unset) | 3 | Full run. Hidi-agili is ~1.1k clips — one epoch under-fits the adapter. |
| `lr_scheduler_type` | linear | cosine | Smoother decay over a real schedule. |
| `warmup_steps` | 5 | `warmup_ratio=0.03` | Scales with run length instead of being a fixed 5 steps. |
| `optim` | `adamw_8bit` | `adamw_torch_fused` | 8-bit optimizer saves VRAM we don't need; fused AdamW is faster on A100+. |
| `bf16` / `fp16` | fp16 unless bf16 | bf16 (A100 supports it) + `tf32=True` | bf16 has wider dynamic range than fp16; tf32 speeds up matmuls. |
| `dataloader_num_workers` | 0 | `cpu_count()//2` | Default 0 starves the GPU during audio-feature loading. |
| `dataloader_pin_memory` | False | True | Faster host→device copies. |
| Eval / save | every 5 steps, no best-model load | every 200 steps, `load_best_model_at_end=True`, `metric_for_best_model="wer"` | Step-5 eval is noise at this batch size; we want the best WER checkpoint, not the last one. |
| Data prep | per-example list comp | `dataset.map(num_proc=cpu_count-2)` | The list comp was single-threaded and dominated wall-clock before training even started. |

`use_gradient_checkpointing="unsloth"` is kept on — even at 80 GB, Unsloth's checkpointing buys headroom for longer audio without measurable speed cost.

## Reverting to the T4 smoke run

If you re-open the notebook on a free Colab T4, only the trainer cell + the LoRA cell need to change back:

```python
# LoRA cell
target_modules = ["q_proj", "v_proj"]

# Trainer cell
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
max_steps = 60
# remove num_train_epochs, warmup_ratio
warmup_steps = 5
optim = "adamw_8bit"
lr_scheduler_type = "linear"
eval_steps = 5
# drop load_best_model_at_end and friends
```

Everything else (bf16 auto-detect, data prep with `num_proc`) is fine on T4 too — `is_bf16_supported()` will just flip to fp16.

## Expected wall-clock

Rough numbers on A100 80 GB, Hidi-agili (~1.1k clips, 6% test split):
- Data prep (`map` with ~14 procs): ~30–60 s.
- 3 epochs at batch 32: ~15–25 min total training.
- Eval + best-model load: adds ~1–2 min.

If you see <50% GPU utilization during training, the bottleneck is the dataloader — bump `dataloader_num_workers` further or `dataset.with_format("torch")` upstream of the trainer.
