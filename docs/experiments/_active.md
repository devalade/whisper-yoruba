# Active trial — working notes

This file is the scratchpad for the trial currently in flight. It's intentionally rougher than a finished entry — chronological notes, commands you actually ran, errors you actually hit, observations as they happen.

**Promotion rule:** when the trial finishes (shipped, failed, or abandoned), rename this file to `YYYY-MM-DD-<short-name>.md`, restructure it against `_template.md`, add a row to the Index in `README.md`, then re-create an empty `_active.md` for the next trial.

---

## Trial: data-mix expansion (test the diversity hypothesis directly)

**Started:** _not yet — drafting plan_
**Motivation:** The 2026-06-09 v2 ft showed a **−23.8 pt** drop on the Hidi-agili held-out split (73.0% → 49.2%) but only a **−0.5 pt** drop on FLEURS yo_ng vs v1 (66.1% → 65.6%). That gap is the classic single-speaker overfit signature: the model is fitting one TTS voice tightly while barely generalizing to real multi-speaker audio. Pushing more epochs on the same corpus won't move FLEURS — it'll widen the train/eval gap further. The 5-epoch hypothesis is therefore parked. The next training axis is **data diversity**.

**Hypothesis:** Adding FLEURS yo_ng train (~2.5k multi-speaker) and Common Voice yo (~few k crowd-sourced) to the training mix will reduce FLEURS WER by **≥5 absolute points** vs v2 (target ≤60% on FLEURS n=200 / beams=5), even at the same 3 epochs.

**Success criterion:**
- FLEURS yo_ng WER ≤ 60% at N=200 / beams=5 (vs v2's 65.6% at N=25 greedy — need full-grade rerun of v2 at the same setting for clean comparison).
- Do not regress the Hidi-agili held-out eval by more than 5 points (51.7% → ≤56.7%); a small regression is expected as the model spreads its capacity across more distributions.
- Convo eval, when recorded, shows improvement on the `identity` bucket (the bucket motivating the whole thesis chapter).

**Cost cap:** ~$8 / ~6 h wall time. Larger budget than v2 because the dataset is bigger (~15k rows vs 9.5k).

### Open design decisions (resolve before launch)

| Decision | Options | Default |
|---|---|---|
| Starting checkpoint | (a) `openai/whisper-large-v3` cold-start (b) `devalade/whisper-large-v3-yoruba-v2` warm-start | (b) warm-start — v2 already gives a 23-pt head start over base on FLEURS, and we save one fine-tune of compute. Risk: single-speaker bias persists. |
| Data mix ratio | Hidi-agili : FLEURS train : Common Voice yo | Default 1:1:1 by row count, capped to ~5k rows each. Random shuffle across the union. |
| Epochs | 3 / 5 | **3** — match v2 first to isolate the data axis as a clean ablation. |
| LoRA r / target modules | r=32 q+v / r=64 q+v / r=32 q+k+v+o | **r=32 q+v** (same as v2) to isolate the data axis. |
| LR + schedule | 1e-5 constant (as v2) / 1e-5 cosine | **1e-5 constant** (same as v2). |
| Target HF repo | `devalade/whisper-large-v3-yoruba-v3` | Do **not** overwrite v1 or v2. |

All defaults above are picked to **vary only the data axis** vs v2 — that's the ablation that goes in the thesis. Later trials can vary epochs, LR schedule, LoRA shape on top of the winning data mix.

### Pre-launch gates (do these first)

- [ ] Verify FLEURS yo_ng train split loads end-to-end and has reasonable text/audio quality. Sample 10 rows.
- [ ] Verify Common Voice yo is accessible on HF (`mozilla-foundation/common_voice_*` — confirm which version has Yoruba). Sample 10 rows.
- [ ] Add the new datasets to `config.FT_DATASETS` and wire up the loader in `scripts/finetune_whisper.py` to concatenate + shuffle. May need a small adapter for column-schema mismatches across the three sources.
- [ ] Re-run FLEURS A/B for v1 and v2 at **N=200 / beams=5** so the v3 comparison has a thesis-grade baseline (not the N=25 / greedy first-pass numbers). Cost: ~3 h locally on MPS — better to run on the same Vast box right before training. Or fold it into the Vast session and run on GPU (~15 min).
- [ ] If recording `data/audio/eval_convo/` is feasible before launch, do it. The convo eval is a stronger thesis signal than FLEURS alone.

### Setup (when launched)

| | |
|---|---|
| Starting checkpoint | _pending decision_ — see above |
| Adapter | LoRA r=32, α=64, target q_proj+v_proj, dropout 0.05 (same as v2) |
| Datasets | `Hidi-agili/yoruba_tts_dataset` + FLEURS yo_ng train + Common Voice yo (~15k rows total) |
| Epochs | 3 |
| Batch size | 12 per device, grad accum 2 (effective batch 24) |
| Learning rate | 1e-5 constant, warmup 50 steps |
| Precision | bf16 |
| Gradient checkpointing | yes (use_reentrant=False) |
| GPU | 1× RTX 5090 32 GB on Vast.ai |
| Target HF repo | `devalade/whisper-large-v3-yoruba-v3` |

### Log

_(empty — fill once launched)_

### Commands actually run

```bash
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

# Thesis-grade v1/v2 baselines first (~15 min on GPU):
make wer-hf N=200 M1_HF_MODEL=devalade/whisper-large-v3-yoruba
make wer-hf N=200 M1_HF_MODEL=devalade/whisper-large-v3-yoruba-v2

# Then training:
tmux new -s ft
make finetune EPOCHS=3 BATCH=12
# Ctrl-B D. nvidia-smi -l 5 in a second ssh — expect ~24-28 GB VRAM, 95%+ util.

# After it finishes — capture metrics first:
python -c "import torch; print(torch.load('models/whisper-yo-lora/training_args.bin', weights_only=False))" > models/whisper-yo-lora/training_args.txt

# Merge + push under -v3:
python -m scripts.merge_lora --push-to-hub --hub-model-id devalade/whisper-large-v3-yoruba-v3
```

### Evaluation (once v3 is on HF)

All numbers at **N=200 / beams=5** for thesis-grade comparison.

| Eval | base | v1 | v2 | **v3 (this trial, data-mix)** | Δ vs v2 |
|---|---|---|---|---|---|
| FLEURS yo_ng WER | _N=200 baseline_ | _N=200 baseline_ | _N=200 baseline_ | _new_ | |
| Hidi-agili held-out WER | — | _from prior runs_ | 49.2% | _new_ | |
| Convo overall WER | — | _from prereqs_ | _from prereqs_ | _new_ | |
| Convo `identity` WER | — | _from prereqs_ | _from prereqs_ | _new_ | |

### Next decision

- ✅ FLEURS drops ≥5 pts without major Hidi-agili regression → ship as M1 backend, promote to `YYYY-MM-DD-data-mix-v3.md`, flip `config.M1_HF_MODEL` to v3. Next trial = combine data-mix + more epochs.
- ⚠️ FLEURS improves but Hidi-agili regresses badly → "catastrophic forgetting on TTS voice" — promote as ⚠️, try re-balancing the mix (more Hidi-agili weight).
- ❌ FLEURS doesn't move → data wasn't the bottleneck either. Promote as ❌, look at LoRA capacity (r=64, more target modules) or full-fine-tune.

---

## When you finish

1. Decide outcome: ✅ shipped / ⚠️ useful but didn't ship / ❌ failed.
2. `cp _active.md YYYY-MM-DD-<short-name>.md`, restructure against `_template.md`, keep the raw log as an appendix if useful.
3. Add a row to the Index in `README.md`.
4. Reset this file to the empty template for the next trial.
5. Commit the doc with the related code/config change so they ship together.
