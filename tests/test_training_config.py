from pathlib import Path

from nlp_project.training.config import load_training_config


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
