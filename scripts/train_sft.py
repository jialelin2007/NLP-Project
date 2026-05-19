#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from nlp_project.training_config import load_training_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT smoke training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--no-deepspeed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_training_config(args.config)
    model_name = args.model_name_or_path or cfg.model_name_or_path
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.logging_dir:
        cfg.logging_dir.mkdir(parents=True, exist_ok=True)

    print(f"model_name_or_path={model_name}")
    print(f"output_dir={cfg.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={"train": str(cfg.train_file), "validation": str(cfg.validation_file)},
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
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
        save_total_limit=cfg.save_total_limit,
        eval_strategy="no",
        logging_dir=str(cfg.logging_dir) if cfg.logging_dir else None,
        max_length=cfg.max_seq_length,
        deepspeed=None if args.no_deepspeed or cfg.deepspeed is None else str(cfg.deepspeed),
        report_to=[],
        run_name=cfg.run_name,
        assistant_only_loss=True,
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
