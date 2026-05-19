#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-assets/models/Qwen3-32B}"
CONFIG="${CONFIG:-configs/training/qwen3_32b_stage1_8gpu_smoke.yaml}"
LOG_DIR="${LOG_DIR:-runs/logs/qwen3_32b_stage1_smoke}"
mkdir -p "$LOG_DIR"

{
  echo "date=$(date -Iseconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "model_dir=$MODEL_DIR"
  echo "config=$CONFIG"
  nvidia-smi
} | tee "$LOG_DIR/preflight.log"

uv run python scripts/models/inspect_local_model.py "$MODEL_DIR" --load-tokenizer | tee "$LOG_DIR/model_inventory.json"
uv run python scripts/data/validate_sft_data.py | tee "$LOG_DIR/sft_validation.log"

uv run torchrun --nproc_per_node=8 scripts/training/train_sft.py \
  --config "$CONFIG" \
  --model-name-or-path "$MODEL_DIR" \
  2>&1 | tee "$LOG_DIR/train.log"
