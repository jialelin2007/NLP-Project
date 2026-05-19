import importlib.util
from pathlib import Path


def test_train_script_is_importable() -> None:
    script = Path("scripts/training/train_sft.py")
    spec = importlib.util.spec_from_file_location("train_sft", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_train_script_keeps_conversational_dataset_for_assistant_loss() -> None:
    source = Path("scripts/training/train_sft.py").read_text(encoding="utf-8")

    assert "assistant_only_loss=True" in source
    assert "dataset_text_field" not in source
    assert ".map(add_text" not in source
