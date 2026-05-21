#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.stage2_collection import (  # noqa: E402
    fetch_text,
    load_candidates,
    sleep_between_requests,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch arXiv HTML for accepted Stage 2 papers.")
    parser.add_argument(
        "--accepted-papers", type=Path, default=Path("data/raw/stage2/accepted_papers.jsonl")
    )
    parser.add_argument("--html-dir", type=Path, default=Path("data/raw/stage2/html"))
    parser.add_argument("--request-sleep", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--no-sleep", action="store_true")
    return parser.parse_args(argv)


def _fetch_one(candidate, *, client: httpx.Client, overwrite: bool, html_dir: Path) -> dict:
    output_path = html_dir / f"{candidate.arxiv_base_id}.html"
    if output_path.is_file() and not overwrite:
        return {
            "paper_id": candidate.arxiv_id,
            "status": "skipped",
            "html_path": str(output_path),
        }
    html_text = fetch_text(candidate.html_url, client=client)
    output_path.write_text(html_text, encoding="utf-8")
    return {
        "paper_id": candidate.arxiv_id,
        "status": "fetched",
        "html_path": str(output_path),
    }


def fetch_html_files(args: argparse.Namespace) -> dict:
    args.html_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.accepted_papers)
    fetched = 0
    skipped = 0
    failed = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            futures = {
                pool.submit(
                    _fetch_one,
                    candidate,
                    client=client,
                    overwrite=args.overwrite,
                    html_dir=args.html_dir,
                ): candidate
                for candidate in candidates
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching HTML",
                unit="paper",
            ):
                candidate = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "fetched":
                        fetched += 1
                    else:
                        skipped += 1
                except httpx.HTTPError as exc:
                    failed.append({"paper_id": candidate.arxiv_id, "error": str(exc)})
                if not args.no_sleep:
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
    return {
        "papers": len(candidates),
        "fetched": fetched,
        "skipped": skipped,
        "failed": failed,
        "html_dir": str(args.html_dir),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fetch_html_files(args)


if __name__ == "__main__":
    main()
