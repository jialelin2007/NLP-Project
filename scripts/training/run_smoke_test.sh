#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/data/validate_sft_data.py
uv run python scripts/training/train_sft.py \
  --config configs/training/qwen3_32b_stage1_smoke.yaml \
  "$@"
