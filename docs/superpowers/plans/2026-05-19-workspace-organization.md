# Workspace Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the NLP project into functional directories while preserving existing CLI and import compatibility.

**Architecture:** Move canonical implementation modules and scripts into functional packages. Keep old script/config/module paths as wrappers or re-exports so current commands and tests continue to work.

**Tech Stack:** Python 3.12, uv, pytest, ruff, shell scripts, Transformers/TRL project utilities.

---

### Task 1: Add Compatibility Test Coverage

**Files:**
- Modify: `tests/test_train_script_import.py`
- Create: `tests/test_compatibility_entrypoints.py`

- [ ] Add tests that assert old script paths and old config paths exist.
- [ ] Add tests that import both canonical functional modules and legacy re-export modules.
- [ ] Run `uv run pytest tests/test_compatibility_entrypoints.py -q` and verify the test fails before migration because canonical functional paths do not exist yet.

### Task 2: Move Python Modules Into Functional Packages

**Files:**
- Create package files under `src/nlp_project/data/`, `src/nlp_project/evaluation/`, `src/nlp_project/models/`, and `src/nlp_project/training/`.
- Replace legacy top-level modules with compatibility re-exports.
- Update internal imports in scripts and tests where canonical imports are clearer.

- [ ] Move `data_processing.py` to `data/processing.py`.
- [ ] Move `sft_format.py` to `data/sft_format.py`.
- [ ] Move `evaluation.py` to `evaluation/metrics.py`.
- [ ] Move `model_inventory.py` to `models/inventory.py`.
- [ ] Move `training_config.py` to `training/config.py`.
- [ ] Add re-export modules at the old paths.
- [ ] Run focused import tests.

### Task 3: Move CLI Scripts Into Functional Folders

**Files:**
- Create scripts under `scripts/data/`, `scripts/evaluation/`, `scripts/models/`, and `scripts/training/`.
- Keep old `scripts/*.py` and `scripts/*.sh` paths as compatibility wrappers.

- [ ] Move data CLIs to `scripts/data/`.
- [ ] Move training CLIs and shell runners to `scripts/training/`.
- [ ] Move evaluation CLI to `scripts/evaluation/`.
- [ ] Move model inspection CLI to `scripts/models/`.
- [ ] Add wrappers that delegate to the new paths with `runpy.run_path` or `bash`.
- [ ] Run shell syntax checks and script import tests.

### Task 4: Move Configs And Large Local Artifacts

**Files and directories:**
- Move DeepSpeed config to `configs/deepspeed/`.
- Move training YAML configs to `configs/training/`.
- Move `models/` to `assets/models/`.
- Move `outputs/` to `runs/`.
- Update default paths and compatibility wrappers.

- [ ] Add `.gitignore` rules for `assets/models/` and `runs/`.
- [ ] Move large directories without deleting them.
- [ ] Keep old config filenames as compatibility YAML files that point to new canonical config paths where practical.
- [ ] Update scripts and README path references.

### Task 5: Readability Cleanup

**Files:**
- Modify canonical Python scripts and modules touched by the move.

- [ ] Replace repeated project-root bootstrapping with a small shared helper where useful.
- [ ] Group constants and helper functions by responsibility.
- [ ] Keep comments limited to non-obvious path compatibility and training choices.
- [ ] Run `uv run ruff check .`.

### Task 6: End-to-End Verification And Commit

**Commands:**
- `uv run ruff check .`
- `uv run pytest`
- `bash -n scripts/run_smoke_test.sh scripts/run_qwen3_32b_smoke.sh scripts/training/run_smoke_test.sh scripts/training/run_qwen3_32b_smoke.sh`
- `uv run python scripts/validate_sft_data.py`
- `uv run python scripts/evaluate.py --limit 5`

- [ ] Inspect `git status --short` and confirm changes are scoped to organization.
- [ ] Commit tracked changes with a clear organization commit message.
