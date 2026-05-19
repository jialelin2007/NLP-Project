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
    logging_dir: Path | None = None
    run_name: str | None = None
    save_total_limit: int | None = None


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
        logging_dir=Path(data["logging_dir"]) if data.get("logging_dir") else None,
        run_name=data.get("run_name"),
        save_total_limit=int(data["save_total_limit"])
        if data.get("save_total_limit") is not None
        else None,
    )
