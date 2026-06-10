#!/usr/bin/env bash
# One-shot environment setup for a freshly-rented Vast.ai GPU box.
#
# Usage (on the Vast box, after `git clone ... && cd whisper-yoruba`):
#
#     bash scripts/setup_vast.sh
#
# The script is idempotent — re-running it on a half-installed box is safe.
# It does NOT run `huggingface-cli login` (interactive) or launch training —
# both are printed as next-steps at the end.

set -euo pipefail

c_red()   { printf "\033[31m%s\033[0m\n" "$*"; }
c_grn()   { printf "\033[32m%s\033[0m\n" "$*"; }
c_ylw()   { printf "\033[33m%s\033[0m\n" "$*"; }
c_blu()   { printf "\033[34m%s\033[0m\n" "$*"; }
banner()  { echo; c_blu "==> $*"; }

# ---------------------------------------------------------------------------
# 0. Sanity — must run from the repo root
# ---------------------------------------------------------------------------
banner "0. Repo + python sanity"
if [[ ! -f config.py || ! -d scripts ]]; then
    c_red "ERROR: run this from the whisper-yoruba repo root (config.py + scripts/ must exist)."
    exit 1
fi
python --version
which python

# ---------------------------------------------------------------------------
# 1. System packages we rely on (tmux for detached training, git for the repo)
# ---------------------------------------------------------------------------
banner "1. System packages"
need_apt=()
for pkg in tmux git curl; do
    if ! command -v "$pkg" >/dev/null 2>&1; then
        need_apt+=("$pkg")
    fi
done
if [[ ${#need_apt[@]} -gt 0 ]]; then
    c_ylw "Installing missing: ${need_apt[*]}"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${need_apt[@]}"
else
    c_grn "All system packages present."
fi

# ---------------------------------------------------------------------------
# 2. Torch + CUDA — the single most common failure mode on Vast.
#    Most Vast base images preinstall a CUDA torch; only reinstall if missing
#    or if it's a CPU-only build that would silently make training useless.
# ---------------------------------------------------------------------------
banner "2. Torch + CUDA check"
torch_ok=0
python - <<'PY' && torch_ok=1 || torch_ok=0
import sys
try:
    import torch
except Exception as e:
    print(f"torch import failed: {e}")
    sys.exit(1)
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(2)
print(f"capability={torch.cuda.get_device_capability()}")
sys.exit(0)
PY

if [[ $torch_ok -ne 1 ]]; then
    c_ylw "Torch missing or CPU-only — installing CUDA build from PyTorch's CDN (fast)."
    pip install --pre --upgrade torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
    # Verify after install
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA still not available after reinstall'; print('CUDA ok:', torch.cuda.get_device_capability())"
else
    c_grn "Torch with CUDA already installed — skipping torch install."
fi

# ---------------------------------------------------------------------------
# 3. Project dependencies — explicit list so we never re-pull torch from PyPI.
#    Tracks requirements.txt minus torch (handled above) and mlx-whisper
#    (Apple-Silicon only — would actually break transformers on Linux).
# ---------------------------------------------------------------------------
banner "3. Python dependencies (excluding torch + mlx)"
pip install --upgrade \
    "transformers>=4.46,<5" \
    "datasets>=2.20.0,<4.0" \
    sentencepiece sacremoses protobuf \
    soundfile librosa \
    "numpy>=1.26.0" "scipy>=1.13.0" \
    jiwer \
    peft accelerate tensorboard \
    huggingface_hub \
    pyyaml tqdm \
    pyarrow

# ---------------------------------------------------------------------------
# 4. Verify the imports the training script actually needs
# ---------------------------------------------------------------------------
banner "4. Verify imports"
python - <<'PY'
import importlib
mods = [
    "torch", "transformers", "datasets", "peft", "accelerate",
    "jiwer", "soundfile", "huggingface_hub", "pyarrow",
]
for m in mods:
    importlib.import_module(m)
    v = getattr(importlib.import_module(m), "__version__", "?")
    print(f"  ok  {m:<18} {v}")
import torch
print(f"\ntorch CUDA: {torch.cuda.is_available()}   capability: {torch.cuda.get_device_capability()}")
print(f"torch CUDA build: {torch.version.cuda}")
PY

# ---------------------------------------------------------------------------
# 5. Final next-steps banner — actual commands to copy-paste
# ---------------------------------------------------------------------------
banner "5. Next steps"
cat <<'EOF'

Environment is ready. Now, in order:

  # a) HuggingFace login (needs Write scope for the v3 push)
  huggingface-cli login

  # b) Dry-run the data mix — fails fast if a source can't resolve
  python -m scripts.finetune_whisper --print-mix

  # c) Thesis-grade baselines for v1 and v2 (~15 min total)
  make wer-hf PYTHON=python N=200 M1_HF_MODEL=devalade/whisper-large-v3-yoruba
  make wer-hf PYTHON=python N=200 M1_HF_MODEL=devalade/whisper-large-v3-yoruba-v2

  # d) Launch training in tmux so an SSH drop doesn't kill it
  tmux new -s ft
  python -m scripts.finetune_whisper --epochs 3 --batch-size 12 --grad-accum 2
  # detach: Ctrl-B then D
  # monitor (in a second ssh): nvidia-smi -l 5

  # e) After training, snapshot args + merge + push v3
  python -c "import torch; print(torch.load('models/whisper-yo-lora/training_args.bin', weights_only=False))" \
      > models/whisper-yo-lora/training_args.txt
  python -m scripts.merge_lora --push-to-hub --hub-model-id devalade/whisper-large-v3-yoruba-v3

  # f) Evaluate v3 at the same N=200/beams=5 grade
  make wer-hf PYTHON=python N=200 M1_HF_MODEL=devalade/whisper-large-v3-yoruba-v3

EOF
c_grn "Done."
