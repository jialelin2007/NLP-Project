import json
from pathlib import Path


def test_zero3_config_delegates_batch_values_to_trainer() -> None:
    config = json.loads(Path("configs/deepspeed/zero3_bf16.json").read_text(encoding="utf-8"))

    assert config["train_micro_batch_size_per_gpu"] == "auto"
    assert config["train_batch_size"] == "auto"
    assert config["gradient_accumulation_steps"] == "auto"
