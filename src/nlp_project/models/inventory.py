from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInventory:
    model_dir: Path
    has_config: bool
    has_tokenizer_config: bool
    has_index: bool
    num_safetensors: int
    total_safetensors_bytes: int
    missing_files: list[str]

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["model_dir"] = str(self.model_dir)
        return data


def inspect_local_model(model_dir: Path) -> ModelInventory:
    safetensors = sorted(model_dir.glob("*.safetensors"))
    missing_files: list[str] = []
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected = sorted(set(index.get("weight_map", {}).values()))
        missing_files.extend(name for name in expected if not (model_dir / name).exists())
    for required in ["config.json", "tokenizer_config.json"]:
        if not (model_dir / required).exists():
            missing_files.append(required)
    return ModelInventory(
        model_dir=model_dir,
        has_config=(model_dir / "config.json").exists(),
        has_tokenizer_config=(model_dir / "tokenizer_config.json").exists(),
        has_index=index_path.exists(),
        num_safetensors=len(safetensors),
        total_safetensors_bytes=sum(path.stat().st_size for path in safetensors),
        missing_files=sorted(missing_files),
    )
