from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nlp_project.data.processing import contains_self_talk


class SFTValidationError(ValueError):
    pass


def validate_sft_record(record: dict[str, Any]) -> None:
    required = {"id", "domain", "split", "metadata", "messages"}
    missing = required - set(record)
    if missing:
        raise SFTValidationError(f"missing fields: {sorted(missing)}")

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise SFTValidationError("messages must contain exactly system, user, assistant")

    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise SFTValidationError(f"unexpected roles: {roles}")

    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SFTValidationError("message content must be non-empty string")
        if contains_self_talk(content):
            raise SFTValidationError("message contains self-talk or think tag")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)
