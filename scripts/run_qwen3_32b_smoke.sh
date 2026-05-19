#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/Qwen3-32B}"
CONFIG="${CONFIG:-configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml}"
LOG_DIR="${LOG_DIR:-outputs/logs/qwen3_32b_stage1_smoke}"
mkdir -p "$LOG_DIR"

{
  echo "date=$(date -Iseconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "model_dir=$MODEL_DIR"
  echo "config=$CONFIG"
  nvidia-smi
} | tee "$LOG_DIR/preflight.log"

uv run python scripts/inspect_model.py "$MODEL_DIR" --load-tokenizer | tee "$LOG_DIR/model_inventory.json"
uv run python scripts/validate_sft_data.py | tee "$LOG_DIR/sft_validation.log"

torchrun --nproc_per_node=8 scripts/train_sft.py \
  --config "$CONFIG" \
  --model-name-or-path "$MODEL_DIR" \
  2>&1 | tee "$LOG_DIR/train.log"
