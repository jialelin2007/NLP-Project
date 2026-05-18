from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sacrebleu


def compute_sacrebleu(predictions: list[str], references: list[str]) -> float:
    return float(sacrebleu.corpus_bleu(predictions, [references], tokenize="zh").score)


def make_sample_record(
    *, example_id: str, source: str, reference: str, prediction: str
) -> dict[str, str]:
    return {
        "id": example_id,
        "source": source,
        "reference": reference,
        "prediction": prediction,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
