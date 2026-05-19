# Qwen3 32B Stage 1 Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the local `models/Qwen3-32B` checkpoint and run a reproducible 8-GPU DeepSpeed ZeRO-3 Stage 1 SFT smoke test on the existing tiny SFT dataset.

**Architecture:** Add a small model-inspection utility, separate 32B smoke config/output paths from the existing 0.6B smoke run, and wrap distributed execution in a restartable shell script that logs environment, git commit, stdout, stderr, and checkpoint paths. Keep raw models and checkpoints ignored by git.

**Tech Stack:** Python 3.12, uv, Hugging Face Transformers, TRL, PyTorch CUDA, DeepSpeed ZeRO-3, torchrun, pytest, ruff.

---

## File Structure

- Create `src/nlp_project/model_inventory.py`: local model directory checks, safetensors shard/index checks, and tokenizer smoke check helpers.
- Create `tests/test_model_inventory.py`: unit tests for complete and incomplete local model directories using temporary fake files.
- Create `scripts/inspect_model.py`: CLI that validates `models/Qwen3-32B`, prints a JSON summary, and optionally loads the tokenizer/config.
- Create `configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml`: dedicated 32B config writing to `outputs/checkpoints/qwen3_32b_stage1_smoke`.
- Modify `src/nlp_project/training_config.py`: add optional `logging_dir`, `save_total_limit`, and `run_name` config fields.
- Modify `scripts/train_sft.py`: pass optional config fields into `SFTConfig`, print resolved model/config paths, and create the output directory before training.
- Create `scripts/run_qwen3_32b_smoke.sh`: runs model inspection, SFT validation, and `torchrun --nproc_per_node=8` training with DeepSpeed.
- Modify `README.md`: document 32B model verification and 8-GPU smoke commands.

## Task 1: Local Model Inventory

**Files:**
- Create: `src/nlp_project/model_inventory.py`
- Create: `tests/test_model_inventory.py`
- Create: `scripts/inspect_model.py`

- [ ] **Step 1: Write failing tests for local model directory inspection**

Create `tests/test_model_inventory.py`:

```python
import json
from pathlib import Path

from nlp_project.model_inventory import inspect_local_model


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_inspect_local_model_accepts_single_safetensors_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3-0.6B"
    model_dir.mkdir()
    write_json(model_dir / "config.json", {"model_type": "qwen3"})
    write_json(model_dir / "tokenizer_config.json", {"model_max_length": 32768})
    (model_dir / "model.safetensors").write_bytes(b"fake")

    summary = inspect_local_model(model_dir)

    assert summary.model_dir == model_dir
    assert summary.has_config is True
    assert summary.has_tokenizer_config is True
    assert summary.num_safetensors == 1
    assert summary.has_index is False
    assert summary.missing_files == []


def test_inspect_local_model_reports_missing_index_shard(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3-32B"
    model_dir.mkdir()
    write_json(model_dir / "config.json", {"model_type": "qwen3"})
    write_json(model_dir / "tokenizer_config.json", {"model_max_length": 32768})
    write_json(
        model_dir / "model.safetensors.index.json",
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}},
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"fake")

    summary = inspect_local_model(model_dir)

    assert summary.has_index is True
    assert "model-00002-of-00002.safetensors" in summary.missing_files
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_model_inventory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nlp_project.model_inventory'`.

- [ ] **Step 3: Implement model inventory helper**

Create `src/nlp_project/model_inventory.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInventory:
    model_dir: Path
    has_config: bool
    has_tokenizer_config: bool
    has_index: bool
    num_safetensors: int
    total_safetensors_bytes: int
    missing_files: list[str]

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["model_dir"] = str(self.model_dir)
        return data


def inspect_local_model(model_dir: Path) -> ModelInventory:
    safetensors = sorted(model_dir.glob("*.safetensors"))
    missing_files: list[str] = []
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected = sorted(set(index.get("weight_map", {}).values()))
        missing_files.extend(name for name in expected if not (model_dir / name).exists())
    for required in ["config.json", "tokenizer_config.json"]:
        if not (model_dir / required).exists():
            missing_files.append(required)
    return ModelInventory(
        model_dir=model_dir,
        has_config=(model_dir / "config.json").exists(),
        has_tokenizer_config=(model_dir / "tokenizer_config.json").exists(),
        has_index=index_path.exists(),
        num_safetensors=len(safetensors),
        total_safetensors_bytes=sum(path.stat().st_size for path in safetensors),
        missing_files=sorted(missing_files),
    )
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/test_model_inventory.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Add inspection CLI**

Create `scripts/inspect_model.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from nlp_project.model_inventory import inspect_local_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--load-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_local_model(args.model_dir)
    result = summary.to_json_dict()
    if args.load_tokenizer:
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        result["model_type"] = config.model_type
        result["vocab_size"] = len(tokenizer)
        result["chat_template_present"] = bool(tokenizer.chat_template)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if summary.missing_files:
        raise SystemExit(f"missing model files: {summary.missing_files}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Inspect local Qwen3-32B**

Run:

```bash
uv run python scripts/inspect_model.py models/Qwen3-32B --load-tokenizer
```

Expected: JSON with `num_safetensors: 17`, `missing_files: []`, `model_type: qwen3`, and `chat_template_present: true`.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/nlp_project/model_inventory.py tests/test_model_inventory.py scripts/inspect_model.py
git commit -m "feat: inspect local model inventory"
```

## Task 2: 32B Smoke Config and Training Script Options

**Files:**
- Create: `configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml`
- Modify: `src/nlp_project/training_config.py`
- Modify: `tests/test_training_config.py`
- Modify: `scripts/train_sft.py`

- [ ] **Step 1: Extend training config test**

Modify `tests/test_training_config.py` to include:

```python
def test_load_training_config_reads_optional_logging_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: models/Qwen3-32B
train_file: data/processed/stage1/sft/tiny_train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: outputs/checkpoints/qwen3_32b_stage1_smoke
logging_dir: outputs/logs/qwen3_32b_stage1_smoke
run_name: qwen3_32b_stage1_smoke
save_total_limit: 1
max_seq_length: 4096
max_steps: 5
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
deepspeed: configs/ds_zero3_bf16.json
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.logging_dir == Path("outputs/logs/qwen3_32b_stage1_smoke")
    assert config.run_name == "qwen3_32b_stage1_smoke"
    assert config.save_total_limit == 1
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_training_config.py -q
```

Expected: FAIL because `TrainingConfig` lacks optional fields.

- [ ] **Step 3: Add optional fields to config loader**

Modify `src/nlp_project/training_config.py`:

```python
logging_dir: Path | None = None
run_name: str | None = None
save_total_limit: int | None = None
```

and in `load_training_config`:

```python
logging_dir=Path(data["logging_dir"]) if data.get("logging_dir") else None,
run_name=data.get("run_name"),
save_total_limit=int(data["save_total_limit"]) if data.get("save_total_limit") is not None else None,
```

- [ ] **Step 4: Pass optional fields into SFTConfig**

Modify `scripts/train_sft.py` `SFTConfig(...)`:

```python
logging_dir=str(cfg.logging_dir) if cfg.logging_dir else None,
run_name=cfg.run_name,
save_total_limit=cfg.save_total_limit,
```

Before loading the model, add:

```python
cfg.output_dir.mkdir(parents=True, exist_ok=True)
if cfg.logging_dir:
    cfg.logging_dir.mkdir(parents=True, exist_ok=True)
print(f"model_name_or_path={model_name}")
print(f"output_dir={cfg.output_dir}")
```

- [ ] **Step 5: Add 32B smoke config**

Create `configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml`:

```yaml
model_name_or_path: models/Qwen3-32B
train_file: data/processed/stage1/sft/tiny_train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: outputs/checkpoints/qwen3_32b_stage1_smoke
logging_dir: outputs/logs/qwen3_32b_stage1_smoke
run_name: qwen3_32b_stage1_smoke
save_total_limit: 1
max_seq_length: 4096
max_steps: 5
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
deepspeed: configs/ds_zero3_bf16.json
```

- [ ] **Step 6: Verify config tests and lint**

Run:

```bash
uv run pytest tests/test_training_config.py -q
uv run ruff check src/nlp_project/training_config.py scripts/train_sft.py tests/test_training_config.py
```

Expected: tests pass and ruff passes.

- [ ] **Step 7: Commit**

Run:

```bash
git add configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml src/nlp_project/training_config.py tests/test_training_config.py scripts/train_sft.py
git commit -m "feat: add qwen3 32b smoke training config"
```

## Task 3: 8-GPU Smoke Runner

**Files:**
- Create: `scripts/run_qwen3_32b_smoke.sh`
- Modify: `README.md`

- [ ] **Step 1: Write shell runner**

Create `scripts/run_qwen3_32b_smoke.sh`:

```bash
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
```

- [ ] **Step 2: Make runner executable**

Run:

```bash
chmod +x scripts/run_qwen3_32b_smoke.sh
```

- [ ] **Step 3: Document runner**

Modify `README.md`:

```markdown
## Qwen3-32B 8-GPU Smoke

Verify local model files:

```bash
uv run python scripts/inspect_model.py models/Qwen3-32B --load-tokenizer
```

Run 8-GPU ZeRO-3 smoke training:

```bash
bash scripts/run_qwen3_32b_smoke.sh
```

Logs are written to `outputs/logs/qwen3_32b_stage1_smoke/`; checkpoints are
written to `outputs/checkpoints/qwen3_32b_stage1_smoke/`.
```

- [ ] **Step 4: Verify shell syntax**

Run:

```bash
bash -n scripts/run_qwen3_32b_smoke.sh
```

Expected: exits 0.

- [ ] **Step 5: Commit**

Run:

```bash
git add scripts/run_qwen3_32b_smoke.sh README.md
git commit -m "feat: add qwen3 32b smoke runner"
```

## Task 4: Execute Preflight and 8-GPU Smoke

**Files:**
- Generated ignored artifacts under `outputs/logs/qwen3_32b_stage1_smoke/`
- Generated ignored artifacts under `outputs/checkpoints/qwen3_32b_stage1_smoke/`

- [ ] **Step 1: Run model inventory**

Run:

```bash
uv run python scripts/inspect_model.py models/Qwen3-32B --load-tokenizer
```

Expected:
- `missing_files` is `[]`.
- `num_safetensors` is `17`.
- `chat_template_present` is `true`.

- [ ] **Step 2: Run SFT validation**

Run:

```bash
uv run python scripts/validate_sft_data.py
```

Expected: `"errors": 0`.

- [ ] **Step 3: Run 8-GPU smoke training**

Run:

```bash
bash scripts/run_qwen3_32b_smoke.sh
```

Expected:
- `torchrun` launches 8 processes.
- DeepSpeed ZeRO-3 initializes.
- Training completes `max_steps: 5`.
- `outputs/checkpoints/qwen3_32b_stage1_smoke/` contains a saved model or checkpoint.
- `outputs/logs/qwen3_32b_stage1_smoke/train.log` contains five loss log entries and no NaN/OOM.

- [ ] **Step 4: If OOM occurs, apply one change only**

If OOM occurs at model init or first step, edit `configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml`:

```yaml
max_seq_length: 2048
gradient_accumulation_steps: 16
```

Then rerun:

```bash
bash scripts/run_qwen3_32b_smoke.sh
```

Commit the config adjustment only if it is required:

```bash
git add configs/qwen3_32b_full_sft_stage1_8gpu_smoke.yaml
git commit -m "fix: lower qwen3 32b smoke memory pressure"
```

## Task 5: Final Verification and Handoff

**Files:**
- Modify: `README.md` only if command notes change during execution.

- [ ] **Step 1: Run final checks**

Run:

```bash
uv run ruff check .
uv run pytest
bash -n scripts/run_qwen3_32b_smoke.sh
uv run python scripts/inspect_model.py models/Qwen3-32B --load-tokenizer
```

Expected:
- Ruff exits 0.
- Pytest exits 0.
- Shell syntax check exits 0.
- Model inventory reports no missing files.

- [ ] **Step 2: Confirm large artifacts are ignored**

Run:

```bash
git check-ignore -v models/Qwen3-32B/model-00001-of-00017.safetensors outputs/checkpoints/qwen3_32b_stage1_smoke
```

Expected: both paths are ignored by `.gitignore`.

- [ ] **Step 3: Commit final documentation changes if any**

Run:

```bash
git status --short
git add README.md
git commit -m "docs: update qwen3 32b smoke notes"
```

Only run the commit if README changed after Task 3.

