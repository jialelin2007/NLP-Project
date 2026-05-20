#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from nlp_project.training.config import (  # noqa: E402
    configure_wandb_environment,
    load_training_config,
    resolve_attention_implementation,
)
from nlp_project.training.data import load_sft_message_datasets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen translation SFT training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--no-deepspeed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_training_config(args.config)
    configure_wandb_environment(cfg)
    model_name = args.model_name_or_path or cfg.model_name_or_path
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.logging_dir:
        cfg.logging_dir.mkdir(parents=True, exist_ok=True)

    print(f"model_name_or_path={model_name}")
    print(f"output_dir={cfg.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_sft_message_datasets(cfg.train_file, cfg.validation_file)
    attn_implementation = resolve_attention_implementation(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
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
        lr_scheduler_type=cfg.lr_scheduler_type or "linear",
        warmup_ratio=cfg.warmup_ratio or 0.0,
        weight_decay=cfg.weight_decay or 0.0,
        max_grad_norm=cfg.max_grad_norm or 1.0,
        logging_steps=cfg.logging_steps or 1,
        save_steps=cfg.save_steps or cfg.max_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if cfg.eval_steps else "no",
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=cfg.load_best_model_at_end or False,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        seed=cfg.seed,
        logging_dir=str(cfg.logging_dir) if cfg.logging_dir else None,
        max_length=cfg.max_seq_length,
        deepspeed=None if args.no_deepspeed or cfg.deepspeed is None else str(cfg.deepspeed),
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        assistant_only_loss=True,
        packing=cfg.packing if cfg.packing is not None else True,
        eval_packing=cfg.eval_packing if cfg.eval_packing is not None else False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


if __name__ == "__main__":
    main()
