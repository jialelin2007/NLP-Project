#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.stage2_html import build_stage2_document  # noqa: E402
from nlp_project.data.stage2_segments import write_stage2_segments  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cleaned Stage 2 arXiv documents and English segments from HTML."
    )
    parser.add_argument(
        "--accepted-papers", type=Path, default=Path("data/raw/stage2/accepted_papers.jsonl")
    )
    parser.add_argument("--html-dir", type=Path, default=Path("data/raw/stage2/html"))
    parser.add_argument(
        "--documents-dir", type=Path, default=Path("data/processed/stage2/documents")
    )
    parser.add_argument(
        "--segments-path", type=Path, default=Path("data/processed/stage2/segments/all.jsonl")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_stage2_segments(
        candidates_path=args.accepted_papers,
        html_dir=args.html_dir,
        documents_dir=args.documents_dir,
        segments_path=args.segments_path,
        build_document=build_stage2_document,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
