#!/usr/bin/env bash
# Sequentially run `make wer-hf` against a list of HF model repos, deleting
# each model's HF hub cache after its run so we never hold more than one
# whisper-large-v3 checkpoint on disk at a time.
#
# Usage:
#   scripts/bench_models.sh [N] <repo> [<repo> ...]
#
# Example:
#   scripts/bench_models.sh 200 \
#     devalade/whisper-large-v3-yoruba \
#     devalade/whisper-large-v3-yoruba-v2 \
#     devalade/whisper-large-v3-yoruba-v3
#
# Notes:
# - Only purges model dirs we explicitly ran (models--<org>--<repo>). Shared
#   artifacts (openai/whisper-large-v3 processor, datasets) are left alone.
# - Output files come from the Makefile: logs/wer_hf_<slug>.jsonl per model.
# - If a run fails mid-way, the script aborts and the just-failed model's
#   cache is left in place to aid debugging.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [N] <repo> [<repo> ...]" >&2
  exit 2
fi

# First arg is N (sample count) iff it's all-digits; otherwise default to 200.
if [[ "$1" =~ ^[0-9]+$ ]]; then
  N="$1"; shift
else
  N=200
fi

if [[ $# -lt 1 ]]; then
  echo "error: need at least one model repo" >&2
  exit 2
fi

HUB_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

echo "=== bench_models.sh ==="
echo "N=$N  models=$*"
echo "hub cache: $HUB_CACHE"
du -sh "$HUB_CACHE" 2>/dev/null || true
echo

for repo in "$@"; do
  slug="${repo//\//--}"          # devalade/foo -> devalade--foo
  cache_dir="$HUB_CACHE/models--$slug"

  echo "----- [$(date '+%H:%M:%S')] BENCH $repo -----"
  make wer-hf N="$N" M="$repo"

  if [[ -d "$cache_dir" ]]; then
    size=$(du -sh "$cache_dir" | cut -f1)
    echo "freeing $cache_dir ($size)"
    rm -rf "$cache_dir"
  else
    echo "no cache dir at $cache_dir (already gone?)"
  fi
  echo
done

echo "=== done at $(date '+%H:%M:%S') ==="
du -sh "$HUB_CACHE" 2>/dev/null || true
