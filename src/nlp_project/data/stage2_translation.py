from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import httpx

from nlp_project.data.processing import SYSTEM_PROMPT, USER_PROMPT_PREFIX, build_sft_record

ErrorClassification = Literal["retryable", "permanent", "fatal"]


class ResponsesTeacherClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        http_client: httpx.Client | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = http_client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._owns_client = http_client is None

    def translate(self, source: str) -> str:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{USER_PROMPT_PREFIX}{source}"},
            ],
        }
        response = self._client.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return extract_responses_text(response.json())

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def extract_responses_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    texts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    text = "\n".join(part.strip() for part in texts if part.strip()).strip()
    if not text:
        raise ValueError("teacher response did not contain output text")
    return text


def make_teacher_sft_record(
    segment: dict[str, Any], *, target: str, teacher_model: str
) -> dict[str, Any]:
    example = {
        "id": segment["id"],
        "source": segment["source"],
        "target": target.strip(),
        "domain": segment["domain"],
        "split": segment["split"],
        "metadata": {
            **segment.get("metadata", {}),
            "teacher_model": teacher_model,
        },
    }
    record = build_sft_record(example)
    record["source"] = example["source"]
    record["target"] = example["target"]
    return record


def select_shard(segment: dict[str, Any], *, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    metadata = segment.get("metadata", {})
    shard_key = str(metadata.get("paper_id") or segment["id"])
    digest = hashlib.sha1(shard_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def classify_teacher_error(exc: BaseException) -> ErrorClassification:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code in {404, 406}:
            return "permanent"
        if status_code == 429 or status_code >= 500:
            return "retryable"
        return "fatal"
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError)):
        return "retryable"
    return "fatal"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_completed_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        for record in iter_jsonl(path):
            completed.add(record["id"])
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
