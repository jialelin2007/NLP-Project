from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def load_train_sft_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts/training/train_sft.py"
    spec = importlib.util.spec_from_file_location("train_sft_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_sft_passes_resume_checkpoint_from_config(tmp_path: Path) -> None:
    train_sft = load_train_sft_module()
    config_path = tmp_path / "config.yaml"
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    train_file.write_text("", encoding="utf-8")
    validation_file.write_text("", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint-1000"
    output_dir = tmp_path / "output"
    logging_dir = tmp_path / "logs"
    config_path.write_text(
        f"""
model_name_or_path: {checkpoint_dir}
train_file: {train_file}
validation_file: {validation_file}
output_dir: {output_dir}
resume_from_checkpoint: {checkpoint_dir}
logging_dir: {logging_dir}
max_seq_length: 4096
max_steps: 1500
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.000003
bf16: true
gradient_checkpointing: true
report_to: []
""",
        encoding="utf-8",
    )
    trainer = MagicMock()

    with (
        patch("sys.argv", ["train_sft.py", "--config", str(config_path)]),
        patch.object(train_sft, "load_tokenizer") as load_tokenizer,
        patch.object(train_sft, "load_sft_message_datasets") as load_datasets,
        patch.object(train_sft, "resolve_attention_implementation", return_value="sdpa"),
        patch.object(train_sft.AutoModelForCausalLM, "from_pretrained"),
        patch.object(train_sft, "SFTTrainer", return_value=trainer),
    ):
        load_tokenizer.return_value.pad_token = "<|endoftext|>"
        load_datasets.return_value = {"train": [], "validation": []}

        train_sft.main()

    trainer.train.assert_called_once_with(resume_from_checkpoint=str(checkpoint_dir))


def test_train_sft_syncs_resume_checkpoint_state_cadence(tmp_path: Path) -> None:
    train_sft = load_train_sft_module()
    config_path = tmp_path / "config.yaml"
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    train_file.write_text("", encoding="utf-8")
    validation_file.write_text("", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoint-1000"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"save_steps": 500, "eval_steps": 100, "logging_steps": 500}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    logging_dir = tmp_path / "logs"
    config_path.write_text(
        f"""
model_name_or_path: {checkpoint_dir}
train_file: {train_file}
validation_file: {validation_file}
output_dir: {output_dir}
resume_from_checkpoint: {checkpoint_dir}
logging_dir: {logging_dir}
max_seq_length: 4096
max_steps: 1500
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.000003
bf16: true
gradient_checkpointing: true
save_steps: 50
eval_steps: 50
logging_steps: 5
report_to: []
""",
        encoding="utf-8",
    )
    trainer = MagicMock()

    with (
        patch("sys.argv", ["train_sft.py", "--config", str(config_path)]),
        patch.object(train_sft, "load_tokenizer") as load_tokenizer,
        patch.object(train_sft, "load_sft_message_datasets") as load_datasets,
        patch.object(train_sft, "resolve_attention_implementation", return_value="sdpa"),
        patch.object(train_sft.AutoModelForCausalLM, "from_pretrained"),
        patch.object(train_sft, "SFTTrainer", return_value=trainer),
    ):
        load_tokenizer.return_value.pad_token = "<|endoftext|>"
        load_datasets.return_value = {"train": [], "validation": []}

        train_sft.main()

    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["save_steps"] == 50
    assert state["eval_steps"] == 50
    assert state["logging_steps"] == 5
