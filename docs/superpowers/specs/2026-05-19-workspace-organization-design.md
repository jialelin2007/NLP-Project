# Workspace Organization Design

## Goal

Reorganize the repository by function so data preparation, training, evaluation,
model inventory, local assets, and run outputs are easier to find while keeping
the existing public commands compatible.

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
  compatibility wrappers at previous script paths
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

## Compatibility Strategy

Existing commands documented in `README.md` remain valid:

- `uv run python scripts/prepare_stage1_data.py`
- `uv run python scripts/validate_sft_data.py`
- `uv run python scripts/evaluate.py`
- `uv run python scripts/inspect_model.py`
- `uv run python scripts/train_sft.py`
- `bash scripts/run_smoke_test.sh`
- `bash scripts/run_qwen3_32b_smoke.sh`

Each old path becomes a thin wrapper around the new functional script. Config
paths also keep old wrapper files where needed, while canonical configs live
under functional subfolders. Default output paths move from `outputs/` to
`runs/`, and old `outputs/` content is moved into `runs/`.

## Code Organization

Top-level modules under `src/nlp_project/` are split into packages:

- `nlp_project.data.processing` and `nlp_project.data.sft_format`
- `nlp_project.evaluation.metrics`
- `nlp_project.models.inventory`
- `nlp_project.training.config`

Legacy import modules such as `nlp_project.data_processing` remain as
compatibility re-exports. This preserves existing tests and user scripts while
making new imports more descriptive.

## Verification

After migration, run:

```bash
uv run ruff check .
uv run pytest
bash -n scripts/run_smoke_test.sh scripts/run_qwen3_32b_smoke.sh
uv run python scripts/validate_sft_data.py
uv run python scripts/evaluate.py --limit 5
```

The smoke training script is syntax-checked but not launched as a full training
run during cleanup because it can consume all GPUs and hundreds of GB of disk.
