# Fine-tuning runbook (end-to-end)

The full workflow: rent GPU → install → smoke test → train → merge → push to HF → use locally.

## Pre-requisites

- HuggingFace account with a **write-permission** token (`https://huggingface.co/settings/tokens`)
- GitHub access to this repo
- Vast.ai account with credit (~$5 covers one run with margin)
- Your local Mac has the project cloned with the conda env at `$HOME/miniforge3/envs/yoruba/` set up via `make install`

## Step 1 — Rent a GPU

See `docs/vast-ai-setup.md` for picking an instance. For this thesis the recipe targets **1× RTX 5090, vastai/base-image:cuda-12.8.1-auto, ≥ 60 GB disk if possible**.

Cost expectation: ~$0.60–1.00/hr × 3 hours = **$2–3 total** for one training run + merge + push.

## Step 2 — SSH in

Update the Makefile defaults (top of file: `GPU_HOST`, `GPU_PORT`) to your instance, then:

```bash
make gpu-ssh
```

Or one-shot:

```bash
make gpu-ssh GPU_HOST=ssh7.vast.ai GPU_PORT=22141
```

## Step 3 — Install everything on the remote box

```bash
# Verify Blackwell-ready torch
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
# Want: 2.6+ and (12, 0). If not, see docs/vast-ai-setup.md step 4.

# Clone + install
git clone https://github.com/devalade/whisper-yoruba.git
cd whisper-yoruba
pip install -r requirements.txt

# HF login (write-permission token)
huggingface-cli login

# Tell make to use the system python (not the local Mac path)
export PYTHON=python
```

## Step 4 — Smoke test

Always run this first. ~10 minutes, ~$0.10. Catches environment issues before burning hours.

```bash
make finetune-smoke
```

Success criteria:
- `trainable params: ~15M || all params: ~1.55B || trainable%: ~1%` line appears
- Training completes 4 steps without errors
- A WER number is printed (the actual value is meaningless on 50 samples — what matters is that it ran)

If anything errors, see `docs/troubleshooting.md`.

## Step 5 — Real training run

```bash
make finetune EPOCHS=3 BATCH=12
```

Detach: `Ctrl-B D`. Reattach: `tmux a`.

Expected:
- ~2–2.5 hours wall-clock on RTX 5090
- VRAM usage ~24–28 GB (watch with `nvidia-smi -l 5` in a second ssh)
- Train loss drops from ~1.8 → ~0.5–0.8 over 3 epochs
- Eval WER drops each epoch
- Final adapter saved to `models/whisper-yo-lora/`

### What to record for the thesis experiment log

Before clearing logs/cache:

```bash
# Final eval metrics
cat models/whisper-yo-lora/trainer_state.json | python -m json.tool | head -100

# Training arguments (reproducibility)
cat models/whisper-yo-lora/training_args.bin   # binary, you'll need to load in python:
python -c "import pickle; print(pickle.load(open('models/whisper-yo-lora/training_args.bin','rb')))"

# TensorBoard logs — copy to your Mac via scp for plots:
# (on your Mac) scp -P PORT -r root@HOST:/workspace/whisper-yoruba/models/whisper-yo-lora/runs ./tb-logs
```

Copy the salient numbers into `docs/experiments/YYYY-MM-DD-<short-name>.md`.

## Step 6 — Merge LoRA + push to HF Hub

If disk is tight, clean up first (see `docs/vast-ai-setup.md` "Cleanup commands"). Then:

```bash
python -m scripts.merge_lora \
    --push-to-hub \
    --hub-model-id devalade/whisper-large-v3-yoruba
```

This:
1. Loads base `openai/whisper-large-v3`
2. Loads your adapter from `models/whisper-yo-lora/`
3. Merges and writes a plain Whisper checkpoint to `models/whisper-yo-merged/`
4. Creates the HF repo if needed
5. Pushes the model + processor

Upload time: 2–5 minutes for ~3 GB depending on upstream bandwidth.

## Step 7 — Use the model in the pipeline on your Mac

On your **local Mac**, edit `config.py`:

```python
M1_HF_MODEL = "devalade/whisper-large-v3-yoruba"
```

Then:

```bash
# Sanity smoke
make test-m1

# Quantitative — compare against the baseline
make wer-hf N=200

# End-to-end demo
make talk-hf       # push-to-talk RAG
make chat-hf       # push-to-talk free-form
```

`modules/m1_asr_hf.py` will pull the model from HF on first use and cache it locally — no code changes needed.

## Step 8 — Tear down the Vast.ai instance

When you're done, **destroy the instance** on the Vast UI. "Stop" only pauses; "Destroy" stops billing entirely. Your model is safe on HF Hub at this point.

## Iterating: subsequent runs

If you want to try a different recipe (different lr, lora_r, more epochs, more data), the cycle is:

1. Rent a new box (or reuse if still running)
2. `git clone` + `pip install` + `export PYTHON=python` + `huggingface-cli login`
3. `make finetune` with whatever CLI overrides
4. `python -m scripts.merge_lora --push-to-hub --hub-model-id devalade/whisper-large-v3-yoruba-v2` (incremented suffix!)
5. Record a new experiment doc

**Important:** version model IDs (`-v2`, `-v3`) so you can compare. Don't overwrite the working baseline.

## When NOT to push to HF

If a run produces a worse WER than the baseline, **don't push**. Just record the experiment log and move on. The HF repo should only ever hold a model you'd actually want to use.
