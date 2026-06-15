# Free Colab fine-tune (T4)

Cheap, reproducible recipe for trying a Yorùbá LoRA fine-tune of `whisper-large-v3` on a **free** Google Colab T4. Intended for quick iteration: smoke runs, hyperparameter sweeps, and dataset experiments before committing to a paid Vast.ai run (see `docs/fine-tuning-runbook.md`).

Notebook: `Whisper.ipynb` at the repo root, adapted from the Unsloth template.

## When to use Colab vs. Vast.ai

| | Colab T4 (free) | Vast.ai RTX 5090 |
|---|---|---|
| Cost | $0 | ~$2–3 / run |
| VRAM | 15 GB | 32 GB |
| Wall-clock for 1 epoch (~1.1k samples) | ~2–3 h | ~50 min |
| Session cap | 12 h, disconnects on idle | unlimited |
| Best for | smoke runs, sanity checks, demos, dataset probes | real training runs that ship to HF Hub |

**Rule of thumb:** if you'd be sad to lose the run, use Vast.ai.

## Step 1 — Open the notebook in Colab

1. Push `Whisper.ipynb` to GitHub (the repo's `main` branch already has it).
2. In Colab: **File → Open notebook → GitHub** → paste the repo URL → pick `Whisper.ipynb`.
3. **Runtime → Change runtime type → T4 GPU**. Confirm:
   ```python
   !nvidia-smi --query-gpu=name,memory.total --format=csv
   ```
   Expected: `Tesla T4, 15360 MiB`.

If Colab assigns CPU or a different GPU, switch runtime and re-check — you can't fine-tune Whisper-large on CPU.

## Step 2 — (Optional) HF auth via Colab Secrets

Only needed if you plan to push the adapter / merged model to HF Hub at the end. The Hidi-agili dataset and the base Whisper model are both public — no token needed just to train.

**Use Colab Secrets, not `login()` paste-in-cell.** Secrets stay out of the `.ipynb` file, persist across sessions, and you don't have to retype the token every time the runtime recycles.

1. In Colab, click the **🔑 key icon** in the left sidebar.
2. **Add new secret**:
   - Name: `HF_TOKEN`
   - Value: a write-permission token from `https://huggingface.co/settings/tokens`
   - Toggle **Notebook access** ON for `Whisper.ipynb`.
3. The auth cell at the top of the notebook reads it automatically via `google.colab.userdata.get("HF_TOKEN")` and calls `huggingface_hub.login()` for you. Expected output: `HF auth: ok`.

If you see `HF auth: skipped`, the secret name or notebook-access toggle is off — fix and re-run that cell.

The same `HF_TOKEN` variable is referenced by the push cells at the bottom (cell-20, cell-22), so once auth is set up you don't have to paste the token anywhere else.

## Step 3 — Smoke run (default)

`Runtime → Run all`. Cell defaults (`max_steps=60`, `batch=1`, `grad_accum=4`) finish in **~7 min** on T4 and prove the pipeline works end-to-end.

Success criteria (mirror of the Vast.ai runbook):
- `Trainable parameters = 31,457,280 of 1,574,947,840 (2.00% trained)` line appears.
- 60 steps complete without OOM.
- A WER number is printed on FLEURS sample (value is meaningless — what matters is that the inference cell ran).

If you OOM, set `load_in_4bit = True` in cell-5.

## Step 4 — Real run on Colab

In the training-args cell:

```python
# max_steps = 60,            # comment out
num_train_epochs = 1,         # uncomment
per_device_train_batch_size = 1,
gradient_accumulation_steps = 8,   # bump for stability since per-device batch is 1
```

Expected on T4 with ~1.1k Hidi-agili samples:
- ~2.5 h wall-clock
- VRAM ~10–12 GB
- Loss drops ~2.0 → ~0.8 over the epoch

**Plug the laptop in and disable screen lock.** Colab's idle-timeout kicks in if the tab loses focus for too long.

## Step 5 — Save the adapter before the session dies

Cell-20 saves locally to `whisper_yoruba_lora/`. Colab's local disk evaporates when the runtime disconnects, so push to HF Hub:

```python
model.push_to_hub("devalade/whisper-yoruba-lora-colab", token = HF_TOKEN)
tokenizer.push_to_hub("devalade/whisper-yoruba-lora-colab", token = HF_TOKEN)
```

For a merged 16-bit checkpoint (the format `M1_HF_MODEL` in `config.py` expects), use cell-22. Note: merging Whisper-large to 16-bit needs ~6 GB free VRAM on top of the loaded model — on a 15 GB T4 it's tight. If it OOMs, merge offline with `scripts/merge_lora.py` after downloading the adapter.

## Step 6 — Pull the model down locally

On your Mac:

```python
# config.py
M1_HF_MODEL = "devalade/whisper-large-v3-yoruba-colab"
```

Then:

```bash
make test-m1
make wer-hf N=200
```

Record the result in `docs/experiments/YYYY-MM-DD-colab-<short-name>.md`.

## Known gotchas

- **`<|yo|>` token.** Whisper-large-v3 has Yorùbá in its language vocabulary, but if you fork to a smaller variant (`tiny`, `base`) it may not — check `tokenizer.tokenizer.added_tokens_decoder` for `<|yo|>` before training.
- **Streamed FLEURS in the inference cell.** First call downloads a shard; takes 20–30 s. Don't time this as "inference latency."
- **Disconnects mid-train.** Colab Free gives no warning. The notebook's `output_dir = "outputs"` writes to `/content/outputs/` which is wiped on disconnect — for partial-progress safety, mount Google Drive and point `output_dir` there.
- **Hidi-agili dataset size.** ~330 MB parquet, ~1.1k clips. Fine on Colab. If you swap in `mozilla-foundation/common_voice_17_0` (`yo`), expect tens of GB — won't fit T4 disk.

## When this approach stops being useful

Move to Vast.ai when:
- You need >1 epoch on the full dataset.
- You're running ablations and want repeatable wall-clock numbers.
- You're producing the checkpoint that goes into the thesis evaluation table (Colab's "any GPU you happen to get" assignment makes timings non-reproducible).
