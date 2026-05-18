from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

SYSTEM_PROMPT = (
    "You are a professional academic translator. Translate English CS/AI paper text "
    "into accurate, fluent, formal Chinese. Preserve technical terms, equations, "
    "citations, code, variable names, and LaTeX syntax. Do not add explanations."
)

USER_PROMPT_PREFIX = "Translate the following English academic text into Chinese:\n\n"

SELF_TALK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"<\s*/?\s*think\s*>",
        r"\b(?:as an ai|i cannot|i can't|i am unable)\b",
        r"作为\s*(?:一个)?\s*ai",
        r"我是\s*(?:一个)?\s*ai",
        r"我无法",
    ]
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def join_text_parts(*parts: Any) -> str:
    cleaned = [normalize_text(part) for part in parts]
    return "\n\n".join(part for part in cleaned if part)


def contains_self_talk(text: str) -> bool:
    return any(pattern.search(text) for pattern in SELF_TALK_PATTERNS)


def is_valid_translation_pair(
    source: str,
    target: str,
    *,
    min_source_chars: int = 12,
    min_target_chars: int = 6,
    max_char_ratio: float = 8.0,
) -> bool:
    source = normalize_text(source)
    target = normalize_text(target)
    if len(source) < min_source_chars or len(target) < min_target_chars:
        return False
    if contains_self_talk(source) or contains_self_talk(target):
        return False
    ratio = max(len(source), len(target)) / max(1, min(len(source), len(target)))
    return ratio <= max_char_ratio


def make_quickmt_example(record: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    source = normalize_text(record.get("en"))
    target = normalize_text(record.get("zh"))
    return {
        "id": f"quickmt_{split}_{index:010d}",
        "source": source,
        "target": target,
        "domain": "general",
        "split": split,
        "metadata": {
            "source_dataset": "quickmt",
            "score": record.get("sco"),
            "paper_id": None,
            "section": None,
        },
    }


def make_csl_example(
    zh_record: dict[str, Any], en_record: dict[str, Any], *, split: str
) -> dict[str, Any]:
    zh_doc_id = zh_record.get("doc_id")
    en_doc_id = en_record.get("doc_id")
    if zh_doc_id != en_doc_id:
        raise ValueError(f"CSL doc_id mismatch: {zh_doc_id!r} != {en_doc_id!r}")

    source = join_text_parts(en_record.get("title"), en_record.get("abstract"))
    target = join_text_parts(zh_record.get("title"), zh_record.get("abstract"))
    return {
        "id": str(zh_doc_id),
        "source": source,
        "target": target,
        "domain": "scientific",
        "split": split,
        "metadata": {
            "source_dataset": "csl",
            "paper_id": zh_doc_id,
            "section": "title_abstract",
            "keywords": zh_record.get("keywords") or [],
            "keywords_en": en_record.get("keywords") or [],
            "category": zh_record.get("category"),
            "category_eng": zh_record.get("category_eng") or en_record.get("category"),
            "discipline": zh_record.get("discipline"),
            "discipline_eng": zh_record.get("discipline_eng") or en_record.get("discipline"),
        },
    }


def build_sft_record(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "domain": example["domain"],
        "split": example["split"],
        "metadata": example["metadata"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PROMPT_PREFIX}{example['source']}"},
            {"role": "assistant", "content": example["target"]},
        ],
    }


def split_by_stable_hash(identifier: str, *, validation_pct: int = 10, test_pct: int = 10) -> str:
    bucket = int(hashlib.sha1(identifier.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < validation_pct:
        return "validation"
    if bucket < validation_pct + test_pct:
        return "test"
    return "train"


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_quickmt_parquet(path: Path) -> Iterator[dict[str, Any]]:
    table = pd.read_parquet(path)
    yield from table.to_dict(orient="records")
