# Troubleshooting — fine-tuning pipeline

Issues hit during the 2026-06-05 fine-tuning run and how each was fixed. Keep this updated as new issues come up.

## Environment & install

### `make: /root/miniforge3/envs/yoruba/bin/python: No such file or directory`

The Makefile defaults `PYTHON` to a path that exists on the developer's Mac but not on a Vast.ai container. Fix on the remote box:

```bash
export PYTHON=python
```

Or pass per-invocation: `make finetune-smoke PYTHON=python`.

### `ImportError: To support decoding audio data, please install 'torchcodec'`

The `datasets` library version 4.x switched audio decoding from `soundfile` to `torchcodec`. The latter doesn't have wheels for `torch 2.12+cu130` yet (as of 2026-06-05).

Fix: pin `datasets<4` in `requirements.txt`. Already done in the repo — `datasets>=2.20.0,<4.0`. The older path uses `soundfile` which is already installed.

If hit on an existing box:
```bash
pip install "datasets<4.0"
```

### `ImportError: libmlx.so: cannot open shared object file`

`mlx-whisper` is in `requirements.txt` for the Mac mlx backend. On Linux it installs a broken Python wrapper around a missing native library, and `transformers`' tensor-type detection (`is_mlx_array`) imports `mlx.core` unconditionally, which then fails.

Fix: platform-marker it as macOS-only. Already in `requirements.txt`:
```
mlx-whisper>=0.4.0; sys_platform == "darwin"
```

If hit on an existing box:
```bash
pip uninstall -y mlx mlx-whisper
```

### `WARNING: Running pip as the 'root' user`

Benign inside a container. `root` is normal inside Docker, the warning is for general-purpose hosts. Ignore.

## PEFT + Whisper compatibility

### `TypeError: WhisperForConditionalGeneration.forward() got an unexpected keyword argument 'input_ids'`

Caused by passing `task_type="SEQ_2_SEQ_LM"` to `LoraConfig`. That task type tells PEFT to prep inputs as a generic encoder-decoder (T5/BART/NLLB) and inject `input_ids`. Whisper takes mel-spectrograms via `input_features` and rejects `input_ids`.

Fix: drop `task_type` from `LoraConfig`. Already done in `scripts/finetune_whisper.py`. PEFT then passes kwargs through unchanged.

### `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

Comes with the warning `None of the inputs have requires_grad=True. Gradients will be None`.

Root cause: PEFT freezes the base model. The **legacy reentrant** `torch.utils.checkpoint` calls `check_backward_validity` which requires at least one input to have `requires_grad=True` — but every input to a checkpointed block in a frozen base lacks it.

Fix: switch to non-reentrant gradient checkpointing, which uses saved-tensor hooks and handles frozen-base + trainable-adapter as a first-class case. In `scripts/finetune_whisper.py`:

```python
training_args = Seq2SeqTrainingArguments(
    ...
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ...
)
```

Already in place. This is the standard pattern in HF's recent PEFT-Whisper recipes.

(Earlier attempt: a forward hook on `encoder.conv1` that re-enables `requires_grad`. This works in principle but is fragile and didn't fire reliably in our environment. Non-reentrant checkpointing replaces it.)

## Generation warnings (benign)

### `The attention mask is not set and cannot be inferred from input because pad token is same as eos token`

Triggered by HF's generic `generate()` check during eval. Doesn't apply to Whisper meaningfully — Whisper's input is mel-spectrograms, not text tokens, and labels are already masked with `-100` for loss. The warning is a false positive specific to Whisper's tokenizer sharing token 50257 for both pad and eos.

Ignore. Eval metrics are correct.

### `Moving the following attributes in the config to the generation config: {'max_length': 448, 'begin_suppress_tokens': [220, 50257]}`

`transformers` auto-migrating legacy fields out of `model.config` and into `model.generation_config`. Cosmetic, no action needed.

## Disk space

### `safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)`

Hit during `merge_lora` writing the merged 3 GB checkpoint to a near-full 32 GB volume.

Diagnosis: check actual cache location with `echo $HF_HOME`. On Vast images it's `/workspace/.hf_home`, **not** `~/.cache/huggingface`. Cleanup commands aimed at `~/.cache/huggingface` will free nothing.

Fix: cleanup commands from `docs/vast-ai-setup.md`. The biggest free-space win is `rm -rf $HF_HOME/datasets` (preprocessed mel features, ~10 GB) and `rm -rf $HF_HOME/hub/datasets--*` (raw audio).

To avoid in future:
- Add `save_total_limit=1` to `Seq2SeqTrainingArguments` (already done) — limits the trainer to keeping only the best checkpoint.
- Pick a Vast instance with 60+ GB disk.

## Connection & tmux

### `sessions should be nested with care, unset $TMUX to force` when running `tmux new`

Vast wraps your shell in a tmux session by default. Don't nest. Just run training directly in the existing session; `Ctrl-B D` detaches and ssh drops won't kill the run. Reattach later with `tmux a`.

## When to add a new entry

Whenever a bug eats more than ~10 minutes to figure out, document it here:
- The error message exactly
- The root cause in one sentence
- The fix (preferably the durable one in the repo, fallback for an existing affected box)
