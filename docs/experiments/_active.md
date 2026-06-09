# Active trial — working notes

This file is the scratchpad for the trial currently in flight. It's intentionally rougher than a finished entry — chronological notes, commands you actually ran, errors you actually hit, observations as they happen.

**Promotion rule:** when the trial finishes (shipped, failed, or abandoned), rename this file to `YYYY-MM-DD-<short-name>.md`, restructure it against `_template.md`, add a row to the Index in `README.md`, then re-create an empty `_active.md` for the next trial.

---

## Trial: 5-epoch ft (test the more-epochs hypothesis directly)

**Started:** _not yet — waiting on FLEURS A/B between v1 and v2 to decide whether to launch_
**Hypothesis:** Loss + eval WER were still monotone-decreasing at the final epoch of the 2026-06-09 3-epoch run (eval loss 1.747 → 0.951 → 0.885; eval WER 73.0% → 51.7% → 49.2%). Training to 5+ epochs should produce a further drop without catastrophic overfit on this single-speaker corpus.
**Success criterion:**
- Beat v2's conversational eval (especially the `identity` bucket) by ≥3 absolute WER points.
- Do not regress FLEURS yo_ng WER vs v2 by more than ~1 absolute point.
- Eval WER continues to drop epoch 3→4→5, or plateaus (not increases).
**Cost cap:** ~$5 / ~4 h wall time. If WER plateaus by epoch 4, stop early via `make finetune EPOCHS=4` next time.

### Pre-launch gates (do these first)

- [ ] Pull artifacts from the 2026-06-09 run to `models/whisper-yo-lora-run-2026-06-09/`.
- [ ] Destroy the current Vast instance (40222150) to stop billing.
- [ ] Set `config.M1_HF_MODEL = "devalade/whisper-large-v3-yoruba-v2"` *temporarily*.
- [ ] `make wer-hf N=200` → record FLEURS yo_ng WER for v2 in `2026-06-09-3epoch-v2.md`.
- [ ] Re-record / refresh `data/audio/eval_convo/` if it doesn't exist.
- [ ] `make wer-convo-hf` → record convo WER for v2 in `2026-06-09-3epoch-v2.md`.
- [ ] Repeat both against `M1_HF_MODEL = "devalade/whisper-large-v3-yoruba"` (v1) to populate the comparison rows.
- [ ] Decide based on results: launch 5-epoch trial, or pivot to data-mix hypothesis instead.

### Setup (when launched)

| | |
|---|---|
| Base model | `openai/whisper-large-v3` |
| Adapter | LoRA r=32, α=64, target q_proj+v_proj, dropout 0.05 |
| Dataset | `Hidi-agili/yoruba_tts_dataset` (9,499 rows) |
| Epochs | **5** (vs 3 in v2) |
| Batch size | 12 per device, grad accum 2 |
| Learning rate | 1e-5 (warmup 50 steps) |
| Precision | bf16 |
| Gradient checkpointing | yes (use_reentrant=False) |
| GPU | 1× RTX 5090 32 GB on Vast.ai |
| Target HF repo | `devalade/whisper-large-v3-yoruba-v3` (do **not** overwrite v1 or v2) |

### Log

_(empty — fill once launched)_

### Commands actually run

```bash
# When ready:
# (Vast: rent new box, vastai/base-image:cuda-12.8.1-auto, 1× RTX 5090)

make gpu-ssh GPU_HOST=... GPU_PORT=...

# On the box:
git clone https://github.com/devalade/whisper-yoruba.git
cd whisper-yoruba
pip install -r requirements.txt

# CRITICAL — verify CUDA torch BEFORE training (the trap that bit v2):
python -c "import torch; print(torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"
# If cuda? False → reinstall:
#   pip uninstall -y torch torchaudio
#   pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

huggingface-cli login
export PYTHON=python
make finetune-smoke    # ~10 min sanity

tmux new -s ft
make finetune EPOCHS=5 BATCH=12
# Ctrl-B D. nvidia-smi -l 5 in a second ssh — expect ~24-28 GB VRAM, 95%+ util.

# After it finishes — capture metrics first:
python -c "import torch; print(torch.load('models/whisper-yo-lora/training_args.bin', weights_only=False))" > models/whisper-yo-lora/training_args.txt

# Merge + push under -v3:
python -m scripts.merge_lora --push-to-hub --hub-model-id devalade/whisper-large-v3-yoruba-v3
```

### Evaluation (once v3 is on HF)

| Eval | v1 | v2 (current ft) | **v3 (this trial, 5 epochs)** | Δ vs v2 |
|---|---|---|---|---|
| FLEURS yo_ng WER (n=200) | _from prereqs_ | _from prereqs_ | _new_ | |
| Convo overall WER | _from prereqs_ | _from prereqs_ | _new_ | |
| Convo `identity` WER | _from prereqs_ | _from prereqs_ | _new_ | |
| Convo `greeting` WER | _from prereqs_ | _from prereqs_ | _new_ | |

### Next decision

- ✅ Beats v2 on convo without FLEURS regression → ship as M1 backend, promote to `YYYY-MM-DD-5epoch-ft.md`, flip `config.M1_HF_MODEL` to v3.
- ⚠️ Beats v2 on the eval split but not on FLEURS/convo → "training-split overfit" — promote as ⚠️, pivot to data-mix hypothesis (add FLEURS train or Common Voice yo).
- ❌ Worse than v2 → epochs were the wrong knob. Promote as ❌, keep v2 as backend.

---

## When you finish

1. Decide outcome: ✅ shipped / ⚠️ useful but didn't ship / ❌ failed.
2. `cp _active.md YYYY-MM-DD-<short-name>.md`, restructure against `_template.md`, keep the raw log as an appendix if useful.
3. Add a row to the Index in `README.md`.
4. Reset this file to the empty template for the next trial.
5. Commit the doc with the related code/config change so they ship together.
