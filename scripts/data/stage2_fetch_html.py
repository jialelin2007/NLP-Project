#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.stage2 import (  # noqa: E402
    fetch_text,
    load_candidates,
    sleep_between_requests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch arXiv HTML for accepted Stage 2 papers.")
    parser.add_argument(
        "--accepted-papers", type=Path, default=Path("data/raw/stage2/accepted_papers.jsonl")
    )
    parser.add_argument("--html-dir", type=Path, default=Path("data/raw/stage2/html"))
    parser.add_argument("--request-sleep", type=float, default=3.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.html_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.accepted_papers)
    fetched = 0
    skipped = 0
    failed = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for candidate in candidates:
            output_path = args.html_dir / f"{candidate.arxiv_base_id}.html"
            if output_path.is_file() and not args.overwrite:
                skipped += 1
                continue
            try:
                html_text = fetch_text(candidate.html_url, client=client)
                output_path.write_text(html_text, encoding="utf-8")
                fetched += 1
            except httpx.HTTPError as exc:
                failed.append({"paper_id": candidate.arxiv_id, "error": str(exc)})
            sleep_between_requests(args.request_sleep)
    print(
        json.dumps(
            {
                "papers": len(candidates),
                "fetched": fetched,
                "skipped": skipped,
                "failed": failed,
                "html_dir": str(args.html_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit(f"Failed to fetch {len(failed)} HTML files.")


if __name__ == "__main__":
    main()
