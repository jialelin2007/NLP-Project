# Workspace Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the NLP project into functional directories with one canonical entry point for each command.

**Architecture:** Move implementation modules, scripts, configs, and run outputs into functional directories. Remove old script/config/module aliases after canonical paths are verified.

**Tech Stack:** Python 3.12, uv, pytest, ruff, shell scripts, Transformers/TRL project utilities.

---

### Task 1: Add Entry Point Test Coverage

**Files:**
- Modify: `tests/test_train_script_import.py`
- Create: `tests/test_compatibility_entrypoints.py`

- [ ] Add tests that assert canonical script paths and canonical config paths exist.
- [ ] Add tests that assert old script paths, old config paths, and legacy re-export modules do not exist.
- [ ] Run `uv run pytest tests/test_compatibility_entrypoints.py -q` and verify the test fails before cleanup if legacy paths still exist.

### Task 2: Move Python Modules Into Functional Packages

**Files:**
- Create package files under `src/nlp_project/data/`, `src/nlp_project/evaluation/`, `src/nlp_project/models/`, and `src/nlp_project/training/`.
- Remove legacy top-level modules after canonical imports are in place.
- Update internal imports in scripts and tests where canonical imports are clearer.

- [ ] Move `data_processing.py` to `data/processing.py`.
- [ ] Move `sft_format.py` to `data/sft_format.py`.
- [ ] Move `evaluation.py` to `evaluation/metrics.py`.
- [ ] Move `model_inventory.py` to `models/inventory.py`.
- [ ] Move `training_config.py` to `training/config.py`.
- [ ] Delete re-export modules at the old paths.
- [ ] Run focused import tests.

### Task 3: Move CLI Scripts Into Functional Folders

**Files:**
- Create scripts under `scripts/data/`, `scripts/evaluation/`, `scripts/models/`, and `scripts/training/`.
- Remove old root `scripts/*.py` and `scripts/*.sh` wrappers.

- [ ] Move data CLIs to `scripts/data/`.
- [ ] Move training CLIs and shell runners to `scripts/training/`.
- [ ] Move evaluation CLI to `scripts/evaluation/`.
- [ ] Move model inspection CLI to `scripts/models/`.
- [ ] Delete wrappers that delegate to the new paths.
- [ ] Run shell syntax checks and script import tests.

### Task 4: Move Configs And Large Local Artifacts

**Files and directories:**
- Move DeepSpeed config to `configs/deepspeed/`.
- Move training YAML configs to `configs/training/`.
- Move `models/` to `assets/models/`.
- Move `outputs/` to `runs/`.
- Update default paths and remove compatibility wrappers.

- [ ] Add `.gitignore` rules for `assets/models/` and `runs/`.
- [ ] Move large directories without deleting them.
- [ ] Delete old config filenames after canonical config paths are referenced everywhere.
- [ ] Update scripts and README path references.

### Task 5: Readability Cleanup

**Files:**
- Modify canonical Python scripts and modules touched by the move.

- [ ] Replace repeated project-root bootstrapping with a small shared helper where useful.
- [ ] Group constants and helper functions by responsibility.
- [ ] Keep comments limited to non-obvious training choices.
- [ ] Run `uv run ruff check .`.

### Task 6: End-to-End Verification And Commit

**Commands:**
- `uv run ruff check .`
- `uv run pytest`
- `bash -n scripts/training/run_smoke_test.sh scripts/training/run_qwen3_32b_smoke.sh`
- `uv run python scripts/data/validate_sft_data.py`
- `uv run python scripts/evaluation/evaluate_translation.py --limit 5 --output-dir /tmp/nlp_project_eval_check`

- [ ] Inspect `git status --short` and confirm changes are scoped to organization.
- [ ] Commit tracked changes with a clear organization commit message.
