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
