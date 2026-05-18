# Stage 1 Smoke Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible tiny end-to-end Stage 1 pipeline from processed SFT JSONL to a few-step Qwen training smoke test and SacreBLEU evaluation.

**Architecture:** Keep the current data preparation script as the source of tiny/dev data, then add three focused layers: tokenizer/data validation, TRL/Transformers SFT smoke training, and generation/evaluation. Training configs live in `configs/`; reusable code lives under `src/nlp_project/`; command entry points live under `scripts/`.

**Tech Stack:** Python 3.12, uv, Hugging Face Transformers, TRL, Datasets, PyTorch CUDA, DeepSpeed config for later scale-up, SacreBLEU, pytest, ruff.

---

## File Structure

- Create `src/nlp_project/sft_format.py`: validates SFT chat JSONL records and estimates tokenizer lengths.
- Create `tests/test_sft_format.py`: unit tests for SFT record validation and tokenizer-length summarization using a fake tokenizer.
- Create `scripts/validate_sft_data.py`: CLI that validates `data/processed/stage1/sft/*.jsonl`, writes `outputs/eval/data_profile/sft_validation.json`, and fails on malformed records.
- Create `configs/ds_zero3_bf16.json`: DeepSpeed ZeRO-3 BF16 config for future full-parameter training.
- Create `configs/qwen3_32b_full_sft_stage1_smoke.yaml`: smoke config that defaults to Qwen/Qwen3-32B but can be overridden to a tiny model for local fast checks.
- Create `src/nlp_project/training_config.py`: typed config loader for the smoke train script.
- Create `tests/test_training_config.py`: validates YAML parsing and default paths.
- Create `scripts/train_sft.py`: minimal TRL SFTTrainer entry point for a few training steps.
- Create `scripts/run_smoke_test.sh`: reproducible one-command tiny training smoke test.
- Create `src/nlp_project/evaluation.py`: generation sample schema, SacreBLEU calculation, and output writers.
- Create `tests/test_evaluation.py`: unit tests for SacreBLEU wrapper and sample output format.
- Create `scripts/evaluate.py`: CLI to generate translations from a model/checkpoint and save metrics/samples.
- Modify `README.md`: add exact validation, smoke train, and evaluation commands.

## Task 1: SFT Data Validation

**Files:**
- Create: `src/nlp_project/sft_format.py`
- Create: `tests/test_sft_format.py`
- Create: `scripts/validate_sft_data.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests for SFT record validation**

Create `tests/test_sft_format.py`:

```python
from nlp_project.sft_format import SFTValidationError, validate_sft_record


def test_validate_sft_record_accepts_three_message_translation_record() -> None:
    record = {
        "id": "x1",
        "domain": "scientific",
        "split": "train",
        "metadata": {"source_dataset": "unit"},
        "messages": [
            {"role": "system", "content": "Translate without explanations."},
            {"role": "user", "content": "Translate:\n\nAn algorithm converges."},
            {"role": "assistant", "content": "算法会收敛。"},
        ],
    }

    validate_sft_record(record)


def test_validate_sft_record_rejects_think_tags() -> None:
    record = {
        "id": "x1",
        "domain": "scientific",
        "split": "train",
        "metadata": {"source_dataset": "unit"},
        "messages": [
            {"role": "system", "content": "Translate."},
            {"role": "user", "content": "Translate:\n\nText"},
            {"role": "assistant", "content": "<think>reasoning</think>译文"},
        ],
    }

    try:
        validate_sft_record(record)
    except SFTValidationError as exc:
        assert "think" in str(exc).lower()
    else:
        raise AssertionError("expected SFTValidationError")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_sft_format.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nlp_project.sft_format'`.

- [ ] **Step 3: Implement SFT validation**

Create `src/nlp_project/sft_format.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nlp_project.data_processing import contains_self_talk


class SFTValidationError(ValueError):
    pass


def validate_sft_record(record: dict[str, Any]) -> None:
    required = {"id", "domain", "split", "metadata", "messages"}
    missing = required - set(record)
    if missing:
        raise SFTValidationError(f"missing fields: {sorted(missing)}")
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise SFTValidationError("messages must contain exactly system, user, assistant")
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise SFTValidationError(f"unexpected roles: {roles}")
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SFTValidationError("message content must be non-empty string")
        if contains_self_talk(content):
            raise SFTValidationError("message contains self-talk or think tag")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/test_sft_format.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Add validation CLI**

Create `scripts/validate_sft_data.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.sft_format import SFTValidationError, iter_jsonl, validate_sft_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/stage1/sft"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/eval/data_profile/sft_validation.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {}
    errors = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        count = 0
        for line_number, record in iter_jsonl(path):
            try:
                validate_sft_record(record)
            except SFTValidationError as exc:
                errors.append({"path": str(path), "line": line_number, "error": str(exc)})
            count += 1
        summary[path.name] = count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"files": summary, "errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(f"SFT validation failed with {len(errors)} errors")
    print(json.dumps({"files": summary, "errors": 0}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Validate current SFT files**

Run:

```bash
uv run python scripts/validate_sft_data.py
```

Expected: JSON output with `tiny_train.jsonl`, `validation.jsonl`, `test.jsonl`, and `"errors": 0`.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/nlp_project/sft_format.py tests/test_sft_format.py scripts/validate_sft_data.py README.md outputs/eval/data_profile/sft_validation.json
git commit -m "feat: validate sft data format"
```

## Task 2: Training Configuration

**Files:**
- Create: `configs/ds_zero3_bf16.json`
- Create: `configs/qwen3_32b_full_sft_stage1_smoke.yaml`
- Create: `src/nlp_project/training_config.py`
- Create: `tests/test_training_config.py`

- [ ] **Step 1: Write failing tests for config loading**

Create `tests/test_training_config.py`:

```python
from pathlib import Path

from nlp_project.training_config import load_training_config


def test_load_training_config_reads_required_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: Qwen/Qwen3-32B
train_file: data/processed/stage1/sft/tiny_train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: outputs/checkpoints/stage1_smoke
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

    assert config.model_name_or_path == "Qwen/Qwen3-32B"
    assert config.train_file == Path("data/processed/stage1/sft/tiny_train.jsonl")
    assert config.max_steps == 5
    assert config.gradient_checkpointing is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_training_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nlp_project.training_config'`.

- [ ] **Step 3: Implement config loader**

Create `src/nlp_project/training_config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    model_name_or_path: str
    train_file: Path
    validation_file: Path
    output_dir: Path
    max_seq_length: int
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    bf16: bool
    gradient_checkpointing: bool
    deepspeed: Path | None = None


def load_training_config(path: Path) -> TrainingConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainingConfig(
        model_name_or_path=data["model_name_or_path"],
        train_file=Path(data["train_file"]),
        validation_file=Path(data["validation_file"]),
        output_dir=Path(data["output_dir"]),
        max_seq_length=int(data["max_seq_length"]),
        max_steps=int(data["max_steps"]),
        per_device_train_batch_size=int(data["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(data["gradient_accumulation_steps"]),
        learning_rate=float(data["learning_rate"]),
        bf16=bool(data["bf16"]),
        gradient_checkpointing=bool(data["gradient_checkpointing"]),
        deepspeed=Path(data["deepspeed"]) if data.get("deepspeed") else None,
    )
```

- [ ] **Step 4: Add configs**

Create `configs/ds_zero3_bf16.json`:

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto"
}
```

Create `configs/qwen3_32b_full_sft_stage1_smoke.yaml`:

```yaml
model_name_or_path: Qwen/Qwen3-32B
train_file: data/processed/stage1/sft/tiny_train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: outputs/checkpoints/stage1_smoke
max_seq_length: 4096
max_steps: 5
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
deepspeed: configs/ds_zero3_bf16.json
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
uv run pytest tests/test_training_config.py -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

Run:

```bash
git add configs/ds_zero3_bf16.json configs/qwen3_32b_full_sft_stage1_smoke.yaml src/nlp_project/training_config.py tests/test_training_config.py
git commit -m "feat: add stage1 smoke training config"
```

## Task 3: SFT Smoke Training Entry Point

**Files:**
- Create: `scripts/train_sft.py`
- Create: `scripts/run_smoke_test.sh`
- Modify: `README.md`

- [ ] **Step 1: Add import-level test**

Create `tests/test_train_script_import.py`:

```python
import importlib.util
from pathlib import Path


def test_train_script_is_importable() -> None:
    script = Path("scripts/train_sft.py")
    spec = importlib.util.spec_from_file_location("train_sft", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/test_train_script_import.py -q
```

Expected: FAIL with missing `scripts/train_sft.py`.

- [ ] **Step 3: Implement training script**

Create `scripts/train_sft.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from nlp_project.training_config import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    return parser.parse_args()


def formatting_func(example):
    messages = example["messages"]
    return messages


def main() -> None:
    args = parse_args()
    cfg = load_training_config(args.config)
    model_name = args.model_name_or_path or cfg.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": str(cfg.train_file), "validation": str(cfg.validation_file)})
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        bf16=cfg.bf16,
        logging_steps=1,
        save_steps=cfg.max_steps,
        eval_strategy="no",
        max_length=cfg.max_seq_length,
        deepspeed=str(cfg.deepspeed) if cfg.deepspeed else None,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add smoke wrapper**

Create `scripts/run_smoke_test.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/validate_sft_data.py
uv run python scripts/train_sft.py \
  --config configs/qwen3_32b_full_sft_stage1_smoke.yaml \
  "$@"
```

Run:

```bash
chmod +x scripts/run_smoke_test.sh
```

- [ ] **Step 5: Run import test**

Run:

```bash
uv run pytest tests/test_train_script_import.py -q
```

Expected: `1 passed`.

- [ ] **Step 6: Run a tiny-model smoke check before Qwen3-32B**

Run:

```bash
bash scripts/run_smoke_test.sh --model-name-or-path Qwen/Qwen3-0.6B
```

Expected: Training completes `max_steps: 5` and writes `outputs/checkpoints/stage1_smoke`.

- [ ] **Step 7: Document commands and commit**

Modify `README.md` to add:

```markdown
## Smoke Training

```bash
uv run python scripts/validate_sft_data.py
bash scripts/run_smoke_test.sh --model-name-or-path Qwen/Qwen3-0.6B
```

For the target full-parameter run, omit the model override after confirming memory:

```bash
bash scripts/run_smoke_test.sh
```
```

Run:

```bash
git add scripts/train_sft.py scripts/run_smoke_test.sh tests/test_train_script_import.py README.md
git commit -m "feat: add sft smoke training entry point"
```

## Task 4: Evaluation CLI

**Files:**
- Create: `src/nlp_project/evaluation.py`
- Create: `tests/test_evaluation.py`
- Create: `scripts/evaluate.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing tests for metric output**

Create `tests/test_evaluation.py`:

```python
from nlp_project.evaluation import compute_sacrebleu, make_sample_record


def test_compute_sacrebleu_returns_score() -> None:
    score = compute_sacrebleu(["算法会收敛。"], ["算法会收敛。"])
    assert score > 99


def test_make_sample_record_preserves_source_reference_prediction() -> None:
    record = make_sample_record(
        example_id="x1",
        source="An algorithm converges.",
        reference="算法会收敛。",
        prediction="算法收敛。",
    )
    assert record["id"] == "x1"
    assert record["source"] == "An algorithm converges."
    assert record["reference"] == "算法会收敛。"
    assert record["prediction"] == "算法收敛。"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/test_evaluation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nlp_project.evaluation'`.

- [ ] **Step 3: Implement evaluation helpers**

Create `src/nlp_project/evaluation.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import sacrebleu


def compute_sacrebleu(predictions: list[str], references: list[str]) -> float:
    return float(sacrebleu.corpus_bleu(predictions, [references], tokenize="zh").score)


def make_sample_record(
    *, example_id: str, source: str, reference: str, prediction: str
) -> dict[str, str]:
    return {
        "id": example_id,
        "source": source,
        "reference": reference,
        "prediction": prediction,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/test_evaluation.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Implement evaluation script**

Create `scripts/evaluate.py` with a baseline mode first:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.evaluation import compute_sacrebleu, make_sample_record, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/stage1/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/copy_source_baseline"))
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = []
    predictions = []
    references = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= args.limit:
                break
            example = json.loads(line)
            prediction = example["source"]
            predictions.append(prediction)
            references.append(example["target"])
            samples.append(
                make_sample_record(
                    example_id=example["id"],
                    source=example["source"],
                    reference=example["target"],
                    prediction=prediction,
                )
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    write_json(args.output_dir / "metrics.json", {"sacrebleu": compute_sacrebleu(predictions, references), "num_samples": len(samples)})
    print(args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run baseline evaluation**

Run:

```bash
uv run python scripts/evaluate.py --limit 100
```

Expected: writes `outputs/eval/copy_source_baseline/metrics.json` and `samples.jsonl`.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/nlp_project/evaluation.py tests/test_evaluation.py scripts/evaluate.py README.md outputs/eval/copy_source_baseline/metrics.json
git commit -m "feat: add baseline translation evaluation"
```

## Task 5: Final Verification and Handoff

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run complete verification**

Run:

```bash
uv run ruff check .
uv run pytest
uv run python scripts/prepare_stage1_data.py
uv run python scripts/validate_sft_data.py
uv run python scripts/evaluate.py --limit 100
```

Expected:
- Ruff exits 0.
- Pytest exits 0.
- Data preparation prints non-zero `tiny_train`, `validation`, `test`.
- SFT validation reports `"errors": 0`.
- Evaluation writes `metrics.json`.

- [ ] **Step 2: Check git state**

Run:

```bash
git status --short
```

Expected: only intended output artifacts are untracked/modified, or no output if all tracked work has been committed.

- [ ] **Step 3: Commit README command updates if needed**

Run:

```bash
git add README.md
git commit -m "docs: document stage1 smoke workflow"
```

Only run this commit if README changed after earlier task commits.

