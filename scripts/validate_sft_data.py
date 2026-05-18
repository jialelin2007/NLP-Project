#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.sft_format import SFTValidationError, iter_jsonl, validate_sft_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SFT chat JSONL files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/stage1/sft"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/eval/data_profile/sft_validation.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, int] = {}
    errors = []

    for path in sorted(args.input_dir.glob("*.jsonl")):
        count = 0
        for line_number, record in iter_jsonl(path):
            try:
                validate_sft_record(record)
            except SFTValidationError as exc:
                errors.append({"path": str(path), "line": line_number, "error": str(exc)})
            count += 1
        summary[path.name] = count

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"files": summary, "errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(f"SFT validation failed with {len(errors)} errors")
    print(json.dumps({"files": summary, "errors": 0}, indent=2))


if __name__ == "__main__":
    main()
