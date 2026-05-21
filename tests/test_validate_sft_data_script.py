from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_validate_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts/data/validate_sft_data.py"
    spec = importlib.util.spec_from_file_location("validate_sft_data_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_sft_data_ignores_non_split_jsonl_outputs(tmp_path: Path, capsys) -> None:
    module = _load_validate_script_module()
    input_dir = tmp_path / "sft"
    input_dir.mkdir()
    valid_record = {
        "id": "x1",
        "domain": "cs_ai_paper",
        "split": "train",
        "metadata": {"source_dataset": "arxiv"},
        "messages": [
            {"role": "system", "content": "Translate."},
            {"role": "user", "content": "Translate:\n\nText"},
            {"role": "assistant", "content": "译文。"},
        ],
    }
    (input_dir / "train.jsonl").write_text(
        json.dumps(valid_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (input_dir / "errors.jsonl").write_text(
        json.dumps({"id": "failed", "error": "gateway"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "validation.json"

    module.main(
        [
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(captured.out)["errors"] == 0
    assert result["files"] == {"train.jsonl": 1}
