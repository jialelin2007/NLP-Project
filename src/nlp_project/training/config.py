from __future__ import annotations

import os
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
    lr_scheduler_type: str | None = None
    warmup_ratio: float | None = None
    weight_decay: float | None = None
    max_grad_norm: float | None = None
    save_steps: int | None = None
    eval_steps: int | None = None
    logging_steps: int | None = None
    load_best_model_at_end: bool | None = None
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    seed: int | None = None
    deepspeed: Path | None = None
    logging_dir: Path | None = None
    run_name: str | None = None
    save_total_limit: int | None = None
    report_to: list[str] | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None


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
        lr_scheduler_type=data.get("lr_scheduler_type"),
        warmup_ratio=float(data["warmup_ratio"]) if data.get("warmup_ratio") is not None else None,
        weight_decay=float(data["weight_decay"]) if data.get("weight_decay") is not None else None,
        max_grad_norm=float(data["max_grad_norm"])
        if data.get("max_grad_norm") is not None
        else None,
        save_steps=int(data["save_steps"]) if data.get("save_steps") is not None else None,
        eval_steps=int(data["eval_steps"]) if data.get("eval_steps") is not None else None,
        logging_steps=int(data["logging_steps"])
        if data.get("logging_steps") is not None
        else None,
        load_best_model_at_end=(
            bool(data["load_best_model_at_end"])
            if data.get("load_best_model_at_end") is not None
            else None
        ),
        metric_for_best_model=data.get("metric_for_best_model"),
        greater_is_better=(
            bool(data["greater_is_better"])
            if data.get("greater_is_better") is not None
            else None
        ),
        seed=int(data["seed"]) if data.get("seed") is not None else None,
        deepspeed=Path(data["deepspeed"]) if data.get("deepspeed") else None,
        logging_dir=Path(data["logging_dir"]) if data.get("logging_dir") else None,
        run_name=data.get("run_name"),
        save_total_limit=int(data["save_total_limit"])
        if data.get("save_total_limit") is not None
        else None,
        report_to=list(data.get("report_to", ["wandb"])),
        wandb_project=data.get("wandb_project"),
        wandb_entity=data.get("wandb_entity"),
        wandb_run_name=data.get("wandb_run_name"),
        wandb_mode=data.get("wandb_mode"),
    )


def configure_wandb_environment(config: TrainingConfig) -> None:
    if not config.report_to or "wandb" not in config.report_to:
        return
    if config.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    if config.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", config.wandb_entity)
    if config.wandb_run_name:
        os.environ.setdefault("WANDB_NAME", config.wandb_run_name)
    elif config.run_name:
        os.environ.setdefault("WANDB_NAME", config.run_name)
    if config.wandb_mode:
        os.environ.setdefault("WANDB_MODE", config.wandb_mode)
