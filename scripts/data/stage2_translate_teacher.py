#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.sft_format import SFTValidationError, validate_sft_record  # noqa: E402
from nlp_project.data.stage2_translation import (  # noqa: E402
    ResponsesTeacherClient,
    append_jsonl,
    classify_teacher_error,
    iter_jsonl,
    make_teacher_sft_record,
    read_completed_ids,
    select_shard,
)


class RequestThrottle:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_seconds = max(0.0, self._next_allowed_at - now)
            self._next_allowed_at = max(now, self._next_allowed_at) + self.min_interval_seconds
        if sleep_seconds:
            time.sleep(sleep_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Stage 2 English paper segments with a teacher model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/stage2/segments/all.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage2/sft"),
    )
    parser.add_argument("--base-url", default="https://api.vip1129.cc/v1")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--request-sleep", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def build_teacher_client(args: argparse.Namespace) -> ResponsesTeacherClient:
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing API key: pass --api-key or set {args.api_key_env}")
    return ResponsesTeacherClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
    )


def _segment_is_in_shard(segment: dict[str, Any], *, num_shards: int, shard_index: int) -> bool:
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard-index must be within [0, num-shards)")
    return select_shard(segment, num_shards=num_shards) == shard_index


def _load_pending_segments(
    args: argparse.Namespace, completed_ids: set[str]
) -> tuple[list[dict], int]:
    pending: list[dict] = []
    skipped_completed = 0
    for segment in iter_jsonl(args.input):
        if not _segment_is_in_shard(
            segment,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        ):
            continue
        if segment["id"] in completed_ids:
            skipped_completed += 1
            continue
        pending.append(segment)
        if args.limit is not None and len(pending) >= args.limit:
            break
    return pending, skipped_completed


def translate_with_retries(
    client: Any,
    segment: dict[str, Any],
    *,
    max_retries: int,
    retry_base_sleep: float,
) -> str:
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.translate(segment["source"])
        except Exception as exc:
            classification = classify_teacher_error(exc)
            if classification != "retryable" or attempt > max_retries:
                raise
            time.sleep(retry_base_sleep * (2 ** (attempt - 1)))


def _translate_one(
    client: Any,
    segment: dict[str, Any],
    *,
    max_retries: int,
    retry_base_sleep: float,
    throttle: RequestThrottle,
) -> dict[str, Any]:
    try:
        throttle.wait()
        target = translate_with_retries(
            client,
            segment,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )
        return {
            "ok": True,
            "record": make_validated_teacher_sft_record(
                segment,
                target=target,
                teacher_model=client.model,
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": {
                "id": segment["id"],
                "split": segment.get("split"),
                "classification": classify_teacher_error(exc),
                "error": str(exc),
            },
        }


def make_validated_teacher_sft_record(
    segment: dict[str, Any], *, target: str, teacher_model: str
) -> dict[str, Any]:
    record = make_teacher_sft_record(segment, target=target, teacher_model=teacher_model)
    try:
        validate_sft_record(record)
    except SFTValidationError as exc:
        raise ValueError(f"teacher output failed SFT validation: {exc}") from exc
    return record


def translate_stage2_segments(args: argparse.Namespace) -> dict[str, int | str]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = read_completed_ids(args.output_dir)
    pending, skipped_completed = _load_pending_segments(args, completed_ids)

    if args.dry_run:
        summary = {
            "pending": len(pending),
            "skipped_completed": skipped_completed,
            "translated": 0,
            "failed": 0,
            "output_dir": str(args.output_dir),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    max_workers = max(1, min(args.max_workers, 5))
    client = build_teacher_client(args)
    throttle = RequestThrottle(args.request_sleep)
    write_lock = threading.Lock()
    translated = 0
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _translate_one,
                    client,
                    segment,
                    max_retries=args.max_retries,
                    retry_base_sleep=args.retry_base_sleep,
                    throttle=throttle,
                ): segment
                for segment in pending
            }
            progress = as_completed(futures)
            if not args.no_progress:
                progress = tqdm(progress, total=len(futures), desc="Teacher translation")
            for future in progress:
                result = future.result()
                with write_lock:
                    if result["ok"]:
                        record = result["record"]
                        append_jsonl(args.output_dir / f"{record['split']}.jsonl", record)
                        translated += 1
                    else:
                        append_jsonl(args.output_dir / "errors.jsonl", result["error"])
                        failed += 1
    finally:
        client.close()

    summary = {
        "pending": len(pending),
        "skipped_completed": skipped_completed,
        "translated": translated,
        "failed": failed,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "state.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    translate_stage2_segments(args)


if __name__ == "__main__":
    main()
