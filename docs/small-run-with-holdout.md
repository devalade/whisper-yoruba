# Fast iteration: small training run with held-out audio

A targeted recipe for shorter feedback loops than the full ~2.5 hour run. Trains on a small slice of the corpus (default 800 samples) with two raw audio clips held out — you transcribe those clips manually after training and check the model against ground truth.

This is **not** the recipe that produces the thesis-quality checkpoint. It's the recipe you use to validate a hypothesis (does changing X help?) before committing to the full run.

## Why hold out audio

The standard training script splits prepared mel features into train/eval. Both halves come from the same upstream distribution and both are scored by the same `compute_metrics`. The eval WER tells you whether the model is *learning*, but it doesn't give you a clip you can sit down and listen to.

The held-out path pops `N` raw audio clips **before** feature extraction and writes them to `data/audio/holdout/` as plain WAV plus a `manifest.jsonl` with the ground-truth Yoruba text. After training, you run the model against those clips locally and judge it yourself — exactly the kind of qualitative check the FLEURS WER number can't give you.

These clips are excluded from both train and eval, so the model has never seen them.

## Run it

On the GPU box (after `pip install` + `huggingface-cli login` per `docs/fine-tuning-runbook.md` steps 1–3):

```bash
make finetune-small               # defaults: 800 total, 2 held out, 20% eval, 3 epochs, batch 8
make finetune-small EPOCHS=5 BATCH=12 HOLDOUT_N=4 TOTAL=1000 EVAL_FRAC=0.15
```

This produces:
- `models/whisper-yo-lora/` — the LoRA adapter
- `data/audio/holdout/holdout_00.wav`, `holdout_01.wav` — held-out audio
- `data/audio/holdout/manifest.jsonl` — `{"id", "wav", "yo", "duration_s"}` per row

Wall time on RTX 5090: ~15–25 minutes for the defaults.

## Merge + push (same as the big run)

```bash
python -m scripts.merge_lora --push-to-hub \
    --hub-model-id devalade/whisper-large-v3-yoruba-small
```

Use a `-small` suffix so it doesn't overwrite a working baseline.

## Pull the holdout WAVs back to your Mac

```bash
# Run from your Mac:
scp -P $GPU_PORT -r root@$GPU_HOST:/workspace/whisper-yoruba/data/audio/holdout \
    data/audio/
```

## Test the model on the held-out clips

```bash
# Switch config.M1_HF_MODEL to your new repo, then:
make test-holdout
```

Output is per-clip: `REF` line, `HYP` line, WER, plus aggregate. Listen to each WAV (open in any audio app) and judge whether the hypothesis is acceptable beyond just the WER number — diacritic placement, word boundaries, plausibility.

For comparison against baseline mlx-whisper:
```bash
python -m scripts.test_holdout --asr mlx
```

## Determinism

The same `--seed` (default 42) reuses the same holdout, train, and eval samples across runs — so you can compare two recipes on identical data. Switch the seed when you want a fresh draw.

## When to graduate to the full run

If the small run shows the change you wanted (e.g. lower eval WER, better holdout transcripts, fewer conversational failures on the eval set in `docs/conversational-eval.md`), then bake the recipe into a full run via `make finetune EPOCHS=3 BATCH=12` and push that as the canonical checkpoint.

If the small run shows no improvement, you saved ~2 hours and ~$2.
