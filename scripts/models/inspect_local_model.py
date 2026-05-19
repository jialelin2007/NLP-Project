#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from nlp_project.models.inventory import inspect_local_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local Hugging Face model directory.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--load-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_local_model(args.model_dir)
    result = summary.to_json_dict()
    if args.load_tokenizer:
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        result["model_type"] = config.model_type
        result["vocab_size"] = len(tokenizer)
        result["chat_template_present"] = bool(tokenizer.chat_template)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if summary.missing_files:
        raise SystemExit(f"missing model files: {summary.missing_files}")


if __name__ == "__main__":
    main()
