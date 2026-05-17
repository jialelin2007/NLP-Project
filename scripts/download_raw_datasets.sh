#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/quickmt-valid.zh-en
mkdir -p data/raw/neuclir-csl
mkdir -p data/raw/quickmt-train.zh-en

uv run hf download quickmt/quickmt-valid.zh-en \
  --repo-type dataset \
  --local-dir data/raw/quickmt-valid.zh-en \
  --max-workers 4

uv run hf download neuclir/csl \
  --repo-type dataset \
  --local-dir data/raw/neuclir-csl \
  --include "data/csl.jsonl.gz" "data/csl.gt.063023.jsonl.gz" "README.md" \
  --max-workers 4

uv run hf download quickmt/quickmt-train.zh-en \
  --repo-type dataset \
  --local-dir data/raw/quickmt-train.zh-en \
  --max-workers 4

