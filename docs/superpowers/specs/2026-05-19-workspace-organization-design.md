# Workspace Organization Design

## Goal

Reorganize the repository by function so data preparation, training, evaluation,
model inventory, local assets, and run outputs are easier to find. The final
workspace uses one canonical location for each entry point; legacy wrapper
paths are intentionally removed.

## Scope

This cleanup covers tracked project code, shell entry points, tests, docs, local
model files, prepared/raw data, and run outputs. It does not delete datasets,
models, checkpoints, or logs. Large ignored files may be moved into clearer
top-level functional directories.

## Target Structure

```text
configs/
  deepspeed/
  training/
assets/
  models/
data/
  raw/
  processed/
  glossary/
runs/
  checkpoints/
  eval/
  logs/
scripts/
  data/
  evaluation/
  models/
  training/
src/nlp_project/
  data/
  evaluation/
  models/
  training/
tests/
  data/
  evaluation/
  models/
  training/
```

## Entry Point Strategy

Only the canonical functional entry points are supported:

- `uv run python scripts/data/prepare_stage1_data.py`
- `uv run python scripts/data/validate_sft_data.py`
- `uv run python scripts/evaluation/evaluate_translation.py`
- `uv run python scripts/models/inspect_local_model.py`
- `uv run python scripts/training/train_sft.py`
- `bash scripts/training/run_smoke_test.sh`
- `bash scripts/training/run_qwen3_32b_smoke.sh`

Canonical configs live under `configs/deepspeed/` and `configs/training/`.
Default output paths move from `outputs/` to `runs/`, and tracked historical
`outputs/` content is moved into `runs/`.

## Code Organization

Top-level modules under `src/nlp_project/` are split into packages:

- `nlp_project.data.processing` and `nlp_project.data.sft_format`
- `nlp_project.evaluation.metrics`
- `nlp_project.models.inventory`
- `nlp_project.training.config`

Legacy import modules such as `nlp_project.data_processing` are removed so new
code has one import path per responsibility.

## Verification

After migration, run:

```bash
uv run ruff check .
uv run pytest
bash -n scripts/training/run_smoke_test.sh scripts/training/run_qwen3_32b_smoke.sh
uv run python scripts/data/validate_sft_data.py
uv run python scripts/evaluation/evaluate_translation.py --limit 5
```

The smoke training script is syntax-checked but not launched as a full training
run during cleanup because it can consume all GPUs and hundreds of GB of disk.
