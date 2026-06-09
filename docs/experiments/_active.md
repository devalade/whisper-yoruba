# Active trial — working notes

This file is the scratchpad for the trial currently in flight. It's intentionally rougher than a finished entry — chronological notes, commands you actually ran, errors you actually hit, observations as they happen.

**Promotion rule:** when the trial finishes (shipped, failed, or abandoned), rename this file to `YYYY-MM-DD-<short-name>.md`, restructure it against `_template.md`, add a row to the Index in `README.md`, then re-create an empty `_active.md` for the next trial.

---

## Trial: more-epochs full run

**Started:** 2026-06-09
**Hypothesis:** 3 epochs on 9,499 single-speaker clips under-fit. Training longer (5+ epochs) will reduce word-boundary errors on short conversational inputs without overfitting catastrophically — the dataset is small but homogeneous, so the risk of memorising it is real but tractable.
**Success criterion:**
- Beat `devalade/whisper-large-v3-yoruba` (current 3-epoch ft) on the conversational eval set, **especially the `identity` bucket** — that's the bucket containing the `Kíni orúkọ rẹ?` failure.
- Do not regress FLEURS yo_ng WER by more than ~1 absolute point vs the current ft.
- Qualitative: the held-out audio clips transcribe with correct word boundaries.
**Cost cap:** ~$5 / ~4 h wall time. If the run isn't producing a checkpoint in that envelope, stop and re-plan.

> ⚠️ **Note on loop choice:** going straight to a full run skips the small-run sanity check that `2026-06-06-small-run-holdout.md` was built for. Justification: the hypothesis is a single-axis change (epochs), not a recipe change, and a small-run with only 800 samples may not exercise enough data per epoch to be informative about over/under-fit. If this feels wrong before launching, run `make finetune-small EPOCHS=5` first and decide based on the held-out clips.

### Setup (as run, not as planned)

| | |
|---|---|
| Base model | `openai/whisper-large-v3` |
| Adapter | LoRA r=32, α=64, target q_proj+v_proj, dropout 0.05 |
| Dataset | `Hidi-agili/yoruba_tts_dataset` (9,499 rows) |
| Epochs | **5** (vs 3 baseline) — _adjust here if you pick differently_ |
| Batch size | 12 per device, grad accum 2 |
| Learning rate | 1e-5 (warmup 50 steps) |
| Precision | bf16 |
| Gradient checkpointing | yes (use_reentrant=False) |
| GPU | 1× RTX 5090 32 GB on Vast.ai |
| Branch / commit | _fill in when you launch_ |
| Target HF repo | `devalade/whisper-large-v3-yoruba-v2` (do **not** overwrite the current baseline) |

### Log

- **YYYY-MM-DD HH:MM** — rented Vast instance, port _____, ssh ok.
- **HH:MM** — `pip install`, `huggingface-cli login`, `export PYTHON=python` done.
- **HH:MM** — `make finetune-smoke` passed / failed (_note_).
- **HH:MM** — launched `make finetune EPOCHS=5 BATCH=12` in tmux.
- **HH:MM** — epoch 1 done, train loss _____, eval WER _____.
- **HH:MM** — epoch 2 done, _____.
- ...

### Commands actually run

```bash
# Local Mac — find a box:
# (Vast UI: vastai/base-image:cuda-12.8.1-auto, 1× RTX 5090, ≥60 GB)

# Local Mac — ssh in:
make gpu-ssh GPU_HOST=ssh?.vast.ai GPU_PORT=????

# On the box:
git clone https://github.com/devalade/whisper-yoruba.git
cd whisper-yoruba
pip install -r requirements.txt
huggingface-cli login
export PYTHON=python

# Sanity:
make finetune-smoke

# The actual run (5 epochs instead of 3):
tmux new -s ft
make finetune EPOCHS=5 BATCH=12
# Ctrl-B D to detach. `tmux a -t ft` to reattach. Watch nvidia-smi in a second ssh.

# After it finishes — capture the numbers BEFORE cleanup:
cat models/whisper-yo-lora/trainer_state.json | python -m json.tool | head -200 > /tmp/trainer_state.txt
python -c "import pickle; print(pickle.load(open('models/whisper-yo-lora/training_args.bin','rb')))" > /tmp/training_args.txt
# scp /tmp/trainer_state.txt and /tmp/training_args.txt back to Mac.

# Merge + push under a -v2 ID so the current baseline survives:
python -m scripts.merge_lora --push-to-hub \
    --hub-model-id devalade/whisper-large-v3-yoruba-v2
```

### Observations

_Fill in as you go. Use this for anything that surprised you — train loss curve shape, eval WER trajectory, VRAM behavior, etc._

### Errors / blockers

- 

### Evaluation (once the model is on HF)

On local Mac, temporarily flip `config.M1_HF_MODEL` to `devalade/whisper-large-v3-yoruba-v2`, then:

```bash
# FLEURS yo_ng — compare against the v1 baseline
make wer-hf N=200

# Conversational eval — the bucket that motivated this trial
make wer-convo-hf

# Held-out audio clips (if you used the small-run path)
make test-holdout
```

Record:

| Eval | v1 (`devalade/whisper-large-v3-yoruba`) | **v2 (this run, 5 epochs)** | Delta |
|---|---|---|---|
| FLEURS yo_ng WER (n=200) | _v1 number_ | _v2 number_ | |
| Convo overall WER | _v1_ | _v2_ | |
| Convo `identity` WER | _v1_ | _v2_ | |
| Convo `greeting` WER | _v1_ | _v2_ | |

### Next decision

- If v2 beats v1 on convo without regressing FLEURS → ship v2, update `config.M1_HF_MODEL`, promote this file to `YYYY-MM-DD-more-epochs-ft.md`, mark ✅ shipped.
- If v2 ties FLEURS but doesn't move convo → epochs weren't the bottleneck. Promote as ⚠️, write up "5 epochs ≈ 3 epochs on this dataset" as a negative result, and pivot to the data-mix hypothesis (adding FLEURS train / Common Voice).
- If v2 regresses → promote as ❌, note over-fit signature (train loss ↓↓ while eval WER ↑), keep v1 as M1 backend.

---

## When you finish

1. Decide outcome: ✅ shipped / ⚠️ useful but didn't ship / ❌ failed.
2. `cp _active.md YYYY-MM-DD-<short-name>.md`, restructure against `_template.md`, keep the raw log as an appendix if useful.
3. Add a row to the Index in `README.md`.
4. Reset this file to the empty template for the next trial.
5. Commit the doc with the related code/config change so they ship together.
