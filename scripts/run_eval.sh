#!/usr/bin/env bash
# Run the evaluation suite with a user-writable HF cache.
# Usage: scripts/run_eval.sh <episode_config.json> <output_dir> [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME=/home/chenguanxu/hf_cache_msbench
export TRANSFORMERS_CACHE=$HF_HOME          # profile points this at a read-only shared dir
export HF_HUB_CACHE=$HF_HOME/hub

exec /home/chenguanxu/venvs/msbench/bin/python -m src.run_pipeline \
    --episode "$1" --output "$2" "${@:3}"
