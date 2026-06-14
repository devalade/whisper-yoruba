# Whisper fine-tune — data-prep gotchas

Two things in the Unsloth Whisper template's data-prep cell are wrong for real runs. Both bit us during the first A100 training attempt. Documented here so the next person doesn't re-discover them.

## Gotcha 1 — never use `num_proc > 1` with HF `Audio` columns on Colab/cloud

**Symptom**: `dataset.map(num_proc=10)` sits at `0%` forever. No error, no progress, no traceback. Workers spin, GPU stays idle, CPU shows N processes alive but doing nothing.

**Cause**: HuggingFace `datasets` multiprocessing for an `Audio`-feature column has to (a) initialise an audio decoder per worker process, and (b) pickle decoded waveforms between processes. On Colab/Vast/most cloud notebooks, this either deadlocks outright or runs slower than single-process because the per-clip pickle cost dominates the feature-extractor work.

**Fix**: keep `num_proc=1`, use `batched=True, batch_size=32`. The HF feature extractor accepts a list of arrays per call, so batched mode gives most of the speedup we wanted from parallelism without any of the fork pain.

```python
train_dataset = dataset["train"].map(
    prepare_batch,
    batched = True,
    batch_size = 32,
    remove_columns = dataset["train"].column_names,
    desc = "Train split",
)
```

For ~9k clips (Hidi-agili current shard), this finishes in 30–60 s on Colab. The "saturate all CPU cores with `num_proc`" intuition from text datasets does *not* transfer to audio datasets.

## Gotcha 2 — don't assume the dataset size

The original notebook's training cell was tuned around the assumption that `Hidi-agili/yoruba_tts_dataset` is ~1.1k clips. The actual shard at the time of the first A100 run was **8,929 clips** — almost 8× larger.

This caused two compound problems:

1. **The single-threaded list-comp prep cell** (the previous gotcha aside) takes ~1.1k× longer to finish at 8,929 examples. With T4-style settings (`batch=1`, `eval_steps=5`), that compounds into hours of just-prep + eval before any real training happens. This is most of why an "80 GB A100 run" of the original notebook ran for 4 hours without finishing one epoch.
2. **`eval_steps=5` becomes ruinous.** At batch 1 grad-accum 4 with 8,929 examples, an epoch is ~2,232 steps → ~446 eval passes, each one decoding the whole eval split. The training run effectively spends >90% of its wall-clock inside `compute_metrics`.

**Fix**:
- The data-prep cell now prints `train=… test=…` after `.map()` finishes. If the printed train size is suspiciously large, stop and recheck assumptions.
- The trainer uses `eval_strategy="epoch"` instead of `eval_steps=5`. Total eval passes equal the epoch count, full stop.
- For sanity, also `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 5` in a separate cell. If GPU utilisation sits below 20%, you're not training — you're prepping or evaling.

## Gotcha 3 — `formatting_prompts_func` was per-example, not batched

The upstream template defines:

```python
def formatting_prompts_func(example):
    audio_arrays = example['audio']['array']
    features = tokenizer.feature_extractor(audio_arrays, sampling_rate=...)
    ...
train_dataset = [formatting_prompts_func(ex) for ex in tqdm(dataset['train'])]
```

The list comprehension calls the feature extractor **once per example**. The feature extractor's Python overhead dominates the C kernel work for short audio. Replacing with a `batched=True` map that processes 32 clips per call is 4–8× faster on the same CPU and avoids materializing the whole dataset as a Python list before training starts.

The new `prepare_batch` is the batched equivalent:

```python
def prepare_batch(batch):
    arrays = [a["array"] for a in batch["audio"]]
    sr     = batch["audio"][0]["sampling_rate"]
    features = feature_extractor(arrays, sampling_rate = sr)
    labels   = text_tokenizer(batch["text"]).input_ids
    return {"input_features": features.input_features, "labels": labels}
```

## Summary — safe defaults for any HF Audio + Whisper feature extractor pipeline

```python
dataset.map(
    prepare_batch,           # batched feature extraction
    batched = True,
    batch_size = 32,         # tune for CPU memory, not parallelism
    num_proc = 1,            # NEVER raise this for Audio columns
    remove_columns = dataset.column_names,
)
```

If you ever truly need to parallelize HF Audio prep, do it *outside* `map` — write a pre-decoded version of the dataset to disk in a one-off script and let `map` operate on numpy arrays, not the `Audio` feature. That bypasses the per-worker decoder init and the pickle-the-waveform problem. We haven't needed it yet.
