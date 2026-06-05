# Vast.ai setup for Blackwell (RTX 5090)

Reproducible recipe for renting a CUDA box on Vast.ai and getting this project running end-to-end. Last validated 2026-06-05 with an RTX 5090 instance.

## 1. Picking the instance

When filtering on the Vast.ai dashboard:

- **GPU:** RTX 5090 (or fall back to L40S, RTX 4090 — see `docs/fine-tuning-runbook.md` for tradeoffs)
- **VRAM ≥ 24 GB** (Whisper-large-v3 LoRA needs ~24 GB with gradient checkpointing)
- **Disk ≥ 60 GB recommended** (the default 32 GB is workable but tight — see "Disk budget" below)
- **CUDA driver ≥ 12.8** (required for Blackwell sm_120)
- **Download bandwidth ≥ 100 Mbps** (model + dataset = ~6 GB to pull)
- **Inet up ≥ 50 Mbps** (you push the merged model to HF Hub, ~3 GB)
- **DLPerf ≥ 30** (Vast's throughput rating — quick proxy for non-degraded cards)

## 2. Docker image

Recommended: **`vastai/base-image:cuda-12.8.1-auto`**

This is a CUDA-only base with Python. PyTorch is *not* pre-installed, which is actually preferred for Blackwell because you control the wheel.

Avoid: generic `pytorch/pytorch` images that may ship Ampere-only wheels and fail at first forward pass on sm_120.

Already-tried-and-works alternative: `vastai/aio-studio:latest` (24 GB) — bigger but includes JupyterLab/VSCode for interactive work. Verify torch capability before training (see step 4).

## 3. SSH in

Vast provides a command like `ssh -p PORT root@sshHOST.vast.ai -L 8080:localhost:8080`. The `-L` is a port forward useful for TensorBoard / Jupyter.

A `make gpu-ssh` target is wired up in the Makefile — override host/port per instance:

```bash
make gpu-ssh GPU_HOST=ssh7.vast.ai GPU_PORT=22141
```

## 4. Install PyTorch with Blackwell support

On the box:

```bash
# Check if torch is already installed and Blackwell-ready
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

Want: `torch 2.6+` (or 2.12+) and capability `(12, 0)`.

If torch is missing or capability is wrong:

```bash
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio
```

(As of 2026-06, the cu128 index serves wheels including newer cu130 builds; both work with sm_120.)

Verify with a bf16 matmul:

```bash
python -c "import torch; x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16); print((x @ x.T).sum())"
```

Should print a number, no errors. Confirms bf16 + cuda kernels actually execute on the 5090.

## 5. Clone + install

```bash
git clone https://github.com/devalade/whisper-yoruba.git
cd whisper-yoruba
pip install -r requirements.txt
huggingface-cli login    # paste a WRITE-permission token from https://huggingface.co/settings/tokens
```

## 6. Tell the Makefile to use the right Python

The local Makefile defaults `PYTHON` to a Mac conda path that doesn't exist on the box. Override:

```bash
export PYTHON=python
```

(Or pass `PYTHON=python` on each `make` invocation.)

## 7. Smoke test (always — proves the stack is correct)

```bash
make finetune-smoke
```

~10 min, trains 50 samples for 1 epoch. Success = a WER number printed at the end without errors. If anything errors, see `docs/troubleshooting.md`.

## 8. Real training run

In the existing tmux session (Vast wraps your shell in one by default — don't nest):

```bash
make finetune EPOCHS=3 BATCH=12
```

Detach with `Ctrl-B D`, reattach later with `tmux a`.

Expected: ~2–2.5 hours on RTX 5090, eval WER printed at the end of each of 3 epochs.

## Disk budget on a 32 GB box

The default Vast disk is 32 GB. The full workflow uses:

| Item | Size | Location |
|---|---|---|
| Base model `whisper-large-v3` | ~3 GB | `$HF_HOME/hub/` (often `/workspace/.hf_home`) |
| Raw dataset | ~3 GB | `$HF_HOME/hub/datasets--*` |
| Preprocessed mel features (after `corpus.map`) | ~10 GB | `$HF_HOME/datasets` |
| Active training checkpoint | ~1 GB | `models/whisper-yo-lora/` |
| Merged checkpoint | ~3 GB | `models/whisper-yo-merged/` |
| **Peak total** | **~20 GB** | |

It fits, but barely. **Find your HF cache location first** — on Vast images it's typically `/workspace/.hf_home`, not `~/.cache/huggingface`:

```bash
echo $HF_HOME
env | grep HF
```

If the run runs out of space mid-merge, this is the culprit. See "Cleanup commands" below.

## Cleanup commands

To free space between runs without losing the trained adapter:

```bash
# 10 GB of preprocessed mel features — not needed for merging
rm -rf "$HF_HOME/datasets"

# Raw dataset cache — only needed if you re-train
rm -rf "$HF_HOME/hub/datasets--"*

# Old training checkpoints (the trainer saves these per epoch)
rm -rf models/whisper-yo-lora/checkpoint-*

# Failed/partial merged outputs
rm -rf models/whisper-yo-merged

# Pip cache
pip cache purge

df -h /workspace
```

After successful merge and HF push, you can also delete the base model cache to free ~3 GB:

```bash
rm -rf "$HF_HOME/hub/models--openai--whisper-large-v3"
```

## Known quirks

- **Vast wraps your shell in tmux by default.** Don't run `tmux new` inside — it errors. Just launch your training command directly; ssh drops won't kill it.
- **`HF_HOME` is set to `/workspace/.hf_home` on most Vast images**, not `~/.cache/huggingface`. Cleanup commands aimed at `~/.cache` will silently free nothing.
- **`pip` runs as root inside the container** — the "running as root" warning is benign; the container is throwaway.
- **`mlx-whisper` from `requirements.txt`** is now marked `; sys_platform == "darwin"` so it won't install on Linux. Earlier versions of `requirements.txt` did install it and broke `transformers`' tensor-type sniffing — see `docs/troubleshooting.md`.
