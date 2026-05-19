import json
from pathlib import Path

from nlp_project.model_inventory import inspect_local_model


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_inspect_local_model_accepts_single_safetensors_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3-0.6B"
    model_dir.mkdir()
    write_json(model_dir / "config.json", {"model_type": "qwen3"})
    write_json(model_dir / "tokenizer_config.json", {"model_max_length": 32768})
    (model_dir / "model.safetensors").write_bytes(b"fake")

    summary = inspect_local_model(model_dir)

    assert summary.model_dir == model_dir
    assert summary.has_config is True
    assert summary.has_tokenizer_config is True
    assert summary.num_safetensors == 1
    assert summary.has_index is False
    assert summary.missing_files == []


def test_inspect_local_model_reports_missing_index_shard(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3-32B"
    model_dir.mkdir()
    write_json(model_dir / "config.json", {"model_type": "qwen3"})
    write_json(model_dir / "tokenizer_config.json", {"model_max_length": 32768})
    write_json(
        model_dir / "model.safetensors.index.json",
        {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        },
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"fake")

    summary = inspect_local_model(model_dir)

    assert summary.has_index is True
    assert "model-00002-of-00002.safetensors" in summary.missing_files
