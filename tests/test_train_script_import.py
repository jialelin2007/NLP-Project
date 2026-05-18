import importlib.util
from pathlib import Path


def test_train_script_is_importable() -> None:
    script = Path("scripts/train_sft.py")
    spec = importlib.util.spec_from_file_location("train_sft", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
