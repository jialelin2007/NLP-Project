#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/validate_sft_data.py
uv run python scripts/train_sft.py \
  --config configs/qwen3_32b_full_sft_stage1_smoke.yaml \
  "$@"
