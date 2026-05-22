#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.stage2_translation import ResponsesTeacherClient  # noqa: E402
from nlp_project.evaluation.metrics import compute_sacrebleu  # noqa: E402

MODELS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
REASONING_EFFORTS = ("low", "medium", "high", "xhigh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare teacher model translations on the first Stage 2 train samples."
    )
    parser.add_argument("--input", type=Path, default=Path("data/processed/stage2/sft/train.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/eval/teacher_model_reasoning_comparison"),
    )
    parser.add_argument("--base-url", default="https://api.vip1129.cc/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=5.0)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_segments(path: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    for record in iter_jsonl(path):
        if record.get("split") != "train":
            continue
        records.append(record)
        if len(records) >= limit:
            break
    return records


def translate_one(
    *,
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    source: str,
    timeout: float,
    max_retries: int,
    retry_base_sleep: float,
) -> str:
    for attempt in range(1, max_retries + 2):
        client = ResponsesTeacherClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
        try:
            return client.translate(source)
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.TransportError,
        ):
            if attempt > max_retries:
                raise
            time.sleep(retry_base_sleep * (2 ** (attempt - 1)))
        finally:
            client.close()
    raise RuntimeError("unreachable retry state")


def score_prediction(prediction: str, reference: str) -> dict[str, float]:
    return {
        "sacrebleu": compute_sacrebleu([prediction], [reference]),
        "length_ratio": len(prediction) / max(1, len(reference)),
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    completed = set()
    for record in iter_jsonl(path):
        completed.add((record["segment_id"], record["model"], record["reasoning_effort"]))
    return completed


def write_markdown(path: Path, records: list[dict]) -> None:
    lines = [
        "# Teacher Model Reasoning Comparison",
        "",
        "| segment | model | effort | SacreBLEU | length ratio | translation |",
        "|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        translation = record["translation"].replace("\n", "<br>")
        row_template = (
            "| {segment_id} | {model} | {effort} | {sacrebleu:.2f} | "
            "{length_ratio:.2f} | {translation} |"
        )
        lines.append(
            row_template.format(
                segment_id=record["segment_id"],
                model=record["model"],
                effort=record["reasoning_effort"],
                sacrebleu=record["metrics"]["sacrebleu"],
                length_ratio=record["metrics"]["length_ratio"],
                translation=translation,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key: pass --api-key or set {args.api_key_env}")

    segments = load_segments(args.input, args.limit)
    output_jsonl = args.output_dir / "translations.jsonl"
    completed_keys = read_completed_keys(output_jsonl)
    results: list[dict] = []
    for segment in segments:
        reference = segment["target"]
        for model in MODELS:
            for reasoning_effort in REASONING_EFFORTS:
                key = (segment["id"], model, reasoning_effort)
                if key in completed_keys:
                    continue
                translation = translate_one(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    source=segment["source"],
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_base_sleep=args.retry_base_sleep,
                )
                record = {
                    "segment_id": segment["id"],
                    "source": segment["source"],
                    "reference": reference,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "translation": translation,
                    "metrics": score_prediction(translation, reference),
                }
                append_jsonl(output_jsonl, record)
                completed_keys.add(key)
                results.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = list(iter_jsonl(output_jsonl))
    write_jsonl(output_jsonl, all_results)
    write_markdown(args.output_dir / "comparison.md", all_results)
    print(args.output_dir)


if __name__ == "__main__":
    main()
