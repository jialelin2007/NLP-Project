#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.stage2 import (  # noqa: E402
    DEFAULT_STAGE2_CATEGORIES,
    deduplicate_latest_arxiv_entries,
    fetch_arxiv_candidates,
    fetch_openalex_work,
    has_usable_arxiv_html,
    make_paper_candidate,
    sleep_between_requests,
    write_candidates,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Stage 2 arXiv CS papers with OpenAlex citation filtering."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/stage2"))
    parser.add_argument("--target-papers", type=int, default=2000)
    parser.add_argument("--min-citations", type=int, default=21)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--request-sleep", type=float, default=3.1)
    parser.add_argument("--openalex-mailto", type=str, default=None)
    parser.add_argument("--skip-html-check", action="store_true")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_STAGE2_CATEGORIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.raw_dir / "arxiv_candidates.jsonl"
    enriched_path = args.raw_dir / "openalex_enriched.jsonl"
    accepted_path = args.raw_dir / "accepted_papers.jsonl"

    accepted = []
    all_entries = []
    enriched_records = []
    seen_base_ids: set[str] = set()

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for start in range(0, args.max_candidates, args.page_size):
            entries = fetch_arxiv_candidates(
                categories=args.categories,
                max_results=args.page_size,
                start=start,
                client=client,
            )
            if not entries:
                break
            sleep_between_requests(args.request_sleep)
            for entry in deduplicate_latest_arxiv_entries(entries):
                if entry.arxiv_base_id in seen_base_ids:
                    continue
                seen_base_ids.add(entry.arxiv_base_id)
                all_entries.append(entry)
                work = fetch_openalex_work(
                    entry.arxiv_base_id,
                    client=client,
                    mailto=args.openalex_mailto,
                )
                sleep_between_requests(max(args.request_sleep, 0.1))
                enriched_record = {
                    "arxiv": asdict(entry),
                    "openalex": asdict(work) if work else None,
                }
                enriched_records.append(enriched_record)
                if work is None or work.cited_by_count < args.min_citations:
                    continue
                candidate = make_paper_candidate(entry, work)
                if not args.skip_html_check and not has_usable_arxiv_html(candidate, client=client):
                    continue
                accepted.append(candidate)
                if len(accepted) >= args.target_papers:
                    break
            print(
                json.dumps(
                    {
                        "seen_candidates": len(all_entries),
                        "accepted": len(accepted),
                        "last_start": start,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if len(accepted) >= args.target_papers:
                break

    write_jsonl(candidates_path, (asdict(entry) for entry in all_entries))
    write_jsonl(enriched_path, enriched_records)
    write_candidates(accepted_path, accepted)
    print(
        json.dumps(
            {
                "arxiv_candidates": len(all_entries),
                "openalex_enriched": len(enriched_records),
                "accepted_papers": len(accepted),
                "accepted_path": str(accepted_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if len(accepted) < args.target_papers:
        raise SystemExit(
            f"Accepted {len(accepted)} papers, below target {args.target_papers}. "
            "Increase --max-candidates or relax filters."
        )


if __name__ == "__main__":
    main()
