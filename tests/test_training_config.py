import os
from pathlib import Path

from nlp_project.training.config import configure_wandb_environment, load_training_config


def test_formal_training_configs_are_present() -> None:
    for config_path in [
        Path("configs/training/qwen3_32b_stage1_full.yaml"),
        Path("configs/training/qwen3_32b_stage2_full.yaml"),
    ]:
        assert config_path.is_file(), config_path
        config = load_training_config(config_path)
        assert config.model_name_or_path
        assert config.output_dir.parts[:2] == ("runs", "checkpoints")
        assert config.deepspeed == Path("configs/deepspeed/zero3_bf16.json")


def test_load_training_config_reads_required_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: Qwen/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_full
max_seq_length: 4096
max_steps: 1000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
deepspeed: configs/deepspeed/zero3_bf16.json
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.model_name_or_path == "Qwen/Qwen3-32B"
    assert config.train_file == Path("data/processed/stage1/sft/train.jsonl")
    assert config.max_steps == 1000
    assert config.gradient_checkpointing is True


def test_load_training_config_reads_optional_logging_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage2/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage2_full
logging_dir: runs/logs/qwen3_32b_stage2_full
run_name: qwen3_32b_stage2_full
save_total_limit: 2
max_seq_length: 4096
max_steps: 1000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
deepspeed: configs/deepspeed/zero3_bf16.json
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.logging_dir == Path("runs/logs/qwen3_32b_stage2_full")
    assert config.run_name == "qwen3_32b_stage2_full"
    assert config.save_total_limit == 2


def test_load_training_config_reads_resume_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_bestval_from1000
resume_from_checkpoint: runs/checkpoints/qwen3_32b_stage1_full/checkpoint-1000
max_seq_length: 4096
max_steps: 1500
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.000003
bf16: true
gradient_checkpointing: true
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.resume_from_checkpoint == Path(
        "runs/checkpoints/qwen3_32b_stage1_full/checkpoint-1000"
    )
    assert config.max_steps == 1500


def test_load_training_config_reads_scheduler_stability_and_eval_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_full
max_seq_length: 4096
max_steps: 4000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.000005
bf16: true
gradient_checkpointing: true
deepspeed: configs/deepspeed/zero3_bf16.json
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.05
max_grad_norm: 1.0
save_steps: 500
eval_steps: 500
logging_steps: 50
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
seed: 42
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.lr_scheduler_type == "cosine"
    assert config.warmup_ratio == 0.03
    assert config.weight_decay == 0.05
    assert config.max_grad_norm == 1.0
    assert config.save_steps == 500
    assert config.eval_steps == 500
    assert config.logging_steps == 50
    assert config.load_best_model_at_end is True
    assert config.metric_for_best_model == "eval_loss"
    assert config.greater_is_better is False
    assert config.seed == 42


def test_load_training_config_defaults_to_wandb_reporting(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_full
max_seq_length: 4096
max_steps: 1000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.report_to == ["wandb"]


def test_load_training_config_reads_wandb_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_full
run_name: qwen3_32b_stage1_full
max_seq_length: 4096
max_steps: 1000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
report_to:
  - wandb
wandb_project: qwen-paper-translation-sft
wandb_entity: paper-translation
wandb_run_name: stage1-final
wandb_mode: online
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.report_to == ["wandb"]
    assert config.wandb_project == "qwen-paper-translation-sft"
    assert config.wandb_entity == "paper-translation"
    assert config.wandb_run_name == "stage1-final"
    assert config.wandb_mode == "online"


def test_load_training_config_reads_speed_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_full
max_seq_length: 3072
max_steps: 4000
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.000005
bf16: true
gradient_checkpointing: true
packing: true
eval_packing: false
attn_implementation: flash_attention_2
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.max_seq_length == 3072
    assert config.per_device_train_batch_size == 2
    assert config.gradient_accumulation_steps == 4
    assert config.packing is True
    assert config.eval_packing is False
    assert config.attn_implementation == "flash_attention_2"


def test_resolve_attention_implementation_falls_back_without_flash_attn(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_full
max_seq_length: 3072
max_steps: 4000
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 0.000005
bf16: true
gradient_checkpointing: true
attn_implementation: flash_attention_2
""",
        encoding="utf-8",
    )
    monkeypatch.setattr("nlp_project.training.config.has_flash_attention_2", lambda: False)

    from nlp_project.training.config import resolve_attention_implementation

    assert resolve_attention_implementation(load_training_config(config_path)) == "sdpa"


def test_configure_wandb_environment_sets_expected_variables(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model_name_or_path: assets/models/Qwen3-32B
train_file: data/processed/stage1/sft/train.jsonl
validation_file: data/processed/stage1/sft/validation.jsonl
output_dir: runs/checkpoints/qwen3_32b_stage1_full
run_name: fallback-run-name
max_seq_length: 4096
max_steps: 1000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00001
bf16: true
gradient_checkpointing: true
report_to:
  - wandb
wandb_project: qwen-paper-translation-sft
wandb_entity: paper-translation
wandb_run_name: stage1-final
wandb_mode: offline
""",
        encoding="utf-8",
    )
    for name in ["WANDB_PROJECT", "WANDB_ENTITY", "WANDB_NAME", "WANDB_MODE"]:
        monkeypatch.delenv(name, raising=False)

    configure_wandb_environment(load_training_config(config_path))

    assert os.environ["WANDB_PROJECT"] == "qwen-paper-translation-sft"
    assert os.environ["WANDB_ENTITY"] == "paper-translation"
    assert os.environ["WANDB_NAME"] == "stage1-final"
    assert os.environ["WANDB_MODE"] == "offline"
