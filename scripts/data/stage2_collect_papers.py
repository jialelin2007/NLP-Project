#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nlp_project.data.stage2_collection import (  # noqa: E402
    DEFAULT_STAGE2_CATEGORIES,
    ArxivEntry,
    PaperCandidate,
    deduplicate_latest_arxiv_entries,
    fetch_arxiv_candidates,
    fetch_openalex_arxiv_works_page,
    fetch_openalex_work,
    has_usable_arxiv_html,
    make_openalex_paper_candidate,
    make_paper_candidate,
    sleep_between_requests,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Stage 2 arXiv CS papers with OpenAlex citation filtering."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/stage2"))
    parser.add_argument(
        "--discovery-source",
        choices=("openalex", "arxiv-api"),
        default="openalex",
    )
    parser.add_argument("--target-papers", type=int, default=2000)
    parser.add_argument("--min-citations", type=int, default=21)
    parser.add_argument("--max-candidates", type=int, default=50000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--request-sleep", type=float, default=0.2)
    parser.add_argument("--http-timeout", type=float, default=60.0)
    parser.add_argument(
        "--checkpoint-every-pages",
        type=int,
        default=1,
        help="Flush incremental checkpoint files after this many processed pages.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--openalex-mailto", type=str, default=None)
    parser.add_argument("--check-html", dest="skip_html_check", action="store_false")
    parser.add_argument("--skip-html-check", dest="skip_html_check", action="store_true")
    parser.set_defaults(skip_html_check=True)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_STAGE2_CATEGORIES))
    parser.add_argument("--openalex-workers", type=int, default=6)
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument(
        "--checkpoint-every-records",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.checkpoint_every_records is not None:
        args.checkpoint_every_pages = max(1, args.checkpoint_every_records)
    return args


def _state_path(raw_dir: Path) -> Path:
    return raw_dir / "collection_state.json"


def _candidates_path(raw_dir: Path) -> Path:
    return raw_dir / "arxiv_candidates.jsonl"


def _enriched_path(raw_dir: Path) -> Path:
    return raw_dir / "openalex_enriched.jsonl"


def _accepted_path(raw_dir: Path) -> Path:
    return raw_dir / "accepted_papers.jsonl"


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_candidates_records(path: Path) -> list[ArxivEntry]:
    return [ArxivEntry(**record) for record in _load_jsonl_records(path)]


def _load_accepted_records(path: Path) -> list[PaperCandidate]:
    return [PaperCandidate(**record) for record in _load_jsonl_records(path)]


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "next_start": 0,
            "pages_completed": 0,
            "last_page_start": 0,
            "accepted_papers": 0,
            "seen_candidates": 0,
            "openalex_cursor": "*",
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "next_start": int(state.get("next_start") or 0),
        "pages_completed": int(state.get("pages_completed") or 0),
        "last_page_start": int(state.get("last_page_start") or 0),
        "accepted_papers": int(state.get("accepted_papers") or 0),
        "seen_candidates": int(state.get("seen_candidates") or 0),
        "openalex_cursor": state.get("openalex_cursor", "*"),
    }


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _append_candidates(path: Path, candidates: list[ArxivEntry]) -> int:
    return _append_jsonl(path, [asdict(candidate) for candidate in candidates])


def _append_paper_candidates(path: Path, candidates: list[PaperCandidate]) -> int:
    return _append_jsonl(path, [asdict(candidate) for candidate in candidates])


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _initial_state(
    *,
    raw_dir: Path,
    resume: bool,
) -> tuple[list[ArxivEntry], list[dict[str, Any]], list[PaperCandidate], dict[str, Any]]:
    if not resume:
        return [], [], [], {
            "next_start": 0,
            "pages_completed": 0,
            "last_page_start": 0,
            "accepted_papers": 0,
            "seen_candidates": 0,
            "openalex_cursor": "*",
        }
    all_entries = _load_candidates_records(_candidates_path(raw_dir))
    enriched_records = _load_jsonl_records(_enriched_path(raw_dir))
    accepted = _load_accepted_records(_accepted_path(raw_dir))
    state = _load_state(_state_path(raw_dir))
    if state["seen_candidates"] < len(all_entries):
        state["seen_candidates"] = len(all_entries)
    if state["accepted_papers"] < len(accepted):
        state["accepted_papers"] = len(accepted)
    if not state.get("openalex_cursor"):
        state["openalex_cursor"] = "*"
    return all_entries, enriched_records, accepted, state


def _write_checkpoint(
    *,
    candidates_path: Path,
    enriched_path: Path,
    accepted_path: Path,
    state_path: Path,
    new_candidates: list[ArxivEntry],
    new_enriched: list[dict[str, Any]],
    new_accepted: list[PaperCandidate],
    state: dict[str, Any],
    final: bool = False,
) -> dict[str, int | str | bool]:
    _append_candidates(candidates_path, new_candidates)
    _append_jsonl(enriched_path, new_enriched)
    _append_paper_candidates(accepted_path, new_accepted)
    _write_state(state_path, state)
    summary = {
        "event": "final_written" if final else "checkpoint_written",
        "new_candidates": len(new_candidates),
        "new_enriched": len(new_enriched),
        "new_accepted": len(new_accepted),
        "next_start": state["next_start"],
        "last_page_start": state["last_page_start"],
        "openalex_cursor": state.get("openalex_cursor"),
        "accepted_path": str(accepted_path),
        "state_path": str(state_path),
        "final": final,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _query_openalex(
    entry: ArxivEntry,
    *,
    client: httpx.Client,
    mailto: str | None,
) -> tuple[ArxivEntry, Any | None]:
    work = fetch_openalex_work(
        entry.arxiv_base_id,
        title=entry.title,
        client=client,
        mailto=mailto,
    )
    return entry, work


def _fetch_page_candidates(
    entries: list[ArxivEntry],
    *,
    client: httpx.Client,
    mailto: str | None,
    openalex_workers: int,
    skip_html_check: bool,
    html_check_client: httpx.Client,
    min_citations: int,
    no_sleep: bool,
    existing_base_ids: set[str],
    progress_callback: Callable[[int], object] | None = None,
) -> tuple[list[ArxivEntry], list[dict[str, Any]], list[PaperCandidate]]:
    page_candidates: list[ArxivEntry] = []
    page_enriched: list[dict[str, Any]] = []
    accepted: list[PaperCandidate] = []
    deduped = [
        entry
        for entry in deduplicate_latest_arxiv_entries(entries)
        if entry.arxiv_base_id not in existing_base_ids
    ]
    if not deduped:
        return page_candidates, page_enriched, accepted

    with ThreadPoolExecutor(max_workers=max(1, openalex_workers)) as pool:
        futures = {
            pool.submit(_query_openalex, entry, client=client, mailto=mailto): entry
            for entry in deduped
        }
        for future in as_completed(futures):
            entry, work = future.result()
            page_candidates.append(entry)
            page_enriched.append(
                {"arxiv": asdict(entry), "openalex": asdict(work) if work else None}
            )
            if progress_callback is not None:
                progress_callback(1)
            if not no_sleep:
                sleep_between_requests(0.0)
            if work is None or work.cited_by_count < min_citations:
                continue
            candidate = make_paper_candidate(entry, work)
            if not skip_html_check and not has_usable_arxiv_html(
                candidate, client=html_check_client
            ):
                continue
            accepted.append(candidate)
    return page_candidates, page_enriched, accepted


def _openalex_records_to_candidates(
    records: list[dict[str, Any]],
    *,
    min_citations: int,
    skip_html_check: bool,
    html_check_client: httpx.Client,
    existing_base_ids: set[str],
    progress_callback: Callable[[int], object] | None = None,
) -> tuple[list[dict[str, Any]], list[PaperCandidate]]:
    enriched: list[dict[str, Any]] = []
    accepted: list[PaperCandidate] = []
    page_seen_base_ids: set[str] = set()
    for record in records:
        enriched.append({"openalex": record})
        candidate = make_openalex_paper_candidate(record, min_citations=min_citations)
        if progress_callback is not None:
            progress_callback(1)
        if candidate is None:
            continue
        if (
            candidate.arxiv_base_id in existing_base_ids
            or candidate.arxiv_base_id in page_seen_base_ids
        ):
            continue
        page_seen_base_ids.add(candidate.arxiv_base_id)
        if not skip_html_check and not has_usable_arxiv_html(candidate, client=html_check_client):
            continue
        accepted.append(candidate)
    return enriched, accepted


def collect_stage2_papers_openalex(
    args: argparse.Namespace,
    *,
    client: httpx.Client | None = None,
) -> dict:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = _candidates_path(args.raw_dir)
    enriched_path = _enriched_path(args.raw_dir)
    accepted_path = _accepted_path(args.raw_dir)
    state_path = _state_path(args.raw_dir)

    _all_entries, enriched_records, accepted, state = _initial_state(
        raw_dir=args.raw_dir,
        resume=not args.no_resume,
    )
    seen_base_ids = {candidate.arxiv_base_id for candidate in accepted}

    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=args.http_timeout, follow_redirects=True)
    html_client = client
    progress = tqdm(
        total=args.max_candidates,
        initial=min(state["seen_candidates"], args.max_candidates),
        desc="Collecting Stage 2 papers",
        unit="work",
        disable=args.no_progress,
    )
    page_since_checkpoint = 0
    try:
        cursor = state.get("openalex_cursor") or "*"
        while (
            cursor is not None
            and len(accepted) < args.target_papers
            and state["seen_candidates"] < args.max_candidates
        ):
            records, next_cursor = fetch_openalex_arxiv_works_page(
                cursor=cursor,
                per_page=args.page_size,
                min_citations=args.min_citations,
                mailto=args.openalex_mailto,
                client=client,
            )
            if not records:
                state["openalex_cursor"] = next_cursor
                break
            page_enriched, page_accepted = _openalex_records_to_candidates(
                records,
                min_citations=args.min_citations,
                skip_html_check=args.skip_html_check,
                html_check_client=html_client,
                existing_base_ids=seen_base_ids,
                progress_callback=progress.update,
            )
            for candidate in page_accepted:
                seen_base_ids.add(candidate.arxiv_base_id)
            enriched_records.extend(page_enriched)
            accepted.extend(page_accepted)
            state["seen_candidates"] += len(records)
            state["accepted_papers"] = len(accepted)
            state["pages_completed"] += 1
            state["openalex_cursor"] = next_cursor
            state["last_page_start"] = state["seen_candidates"]
            print(
                json.dumps(
                    {
                        "event": "openalex_page_completed",
                        "page_seen": len(records),
                        "seen_candidates": state["seen_candidates"],
                        "accepted_papers": len(accepted),
                        "next_cursor": next_cursor,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.no_sleep:
                sleep_between_requests(args.request_sleep)
            page_since_checkpoint += 1
            if (
                args.checkpoint_every_pages > 0
                and page_since_checkpoint >= args.checkpoint_every_pages
            ):
                _write_checkpoint(
                    candidates_path=candidates_path,
                    enriched_path=enriched_path,
                    accepted_path=accepted_path,
                    state_path=state_path,
                    new_candidates=[],
                    new_enriched=page_enriched,
                    new_accepted=page_accepted,
                    state=state,
                )
                page_since_checkpoint = 0
            cursor = next_cursor
    finally:
        progress.close()
        _write_checkpoint(
            candidates_path=candidates_path,
            enriched_path=enriched_path,
            accepted_path=accepted_path,
            state_path=state_path,
            new_candidates=[],
            new_enriched=[],
            new_accepted=[],
            state=state,
            final=True,
        )
        if close_client:
            client.close()

    return {
        "arxiv_candidates": 0,
        "openalex_enriched": len(enriched_records),
        "accepted_papers": len(accepted),
        "accepted_path": str(accepted_path),
        "state_path": str(state_path),
    }


def collect_stage2_papers(args: argparse.Namespace, *, client: httpx.Client | None = None) -> dict:
    if args.discovery_source == "openalex":
        return collect_stage2_papers_openalex(args, client=client)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = _candidates_path(args.raw_dir)
    enriched_path = _enriched_path(args.raw_dir)
    accepted_path = _accepted_path(args.raw_dir)
    state_path = _state_path(args.raw_dir)

    all_entries, enriched_records, accepted, state = _initial_state(
        raw_dir=args.raw_dir,
        resume=not args.no_resume,
    )
    seen_base_ids = {entry.arxiv_base_id for entry in all_entries}

    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=args.http_timeout, follow_redirects=True)
    html_client = client
    progress = tqdm(
        total=args.max_candidates,
        initial=min(state["next_start"], args.max_candidates),
        desc="Collecting Stage 2 papers",
        unit="candidate",
        disable=args.no_progress,
    )
    page_since_checkpoint = 0
    try:
        for start in range(state["next_start"], args.max_candidates, args.page_size):
            entries = fetch_arxiv_candidates(
                categories=args.categories,
                max_results=args.page_size,
                start=start,
                client=client,
            )
            if not entries:
                state["last_page_start"] = start
                state["next_start"] = start + args.page_size
                break
            page_candidates, page_enriched, page_accepted = _fetch_page_candidates(
                entries,
                client=client,
                mailto=args.openalex_mailto,
                openalex_workers=args.openalex_workers,
                skip_html_check=args.skip_html_check,
                html_check_client=html_client,
                min_citations=args.min_citations,
                no_sleep=args.no_sleep,
                existing_base_ids=seen_base_ids,
                progress_callback=progress.update,
            )
            for entry in page_candidates:
                seen_base_ids.add(entry.arxiv_base_id)
            all_entries.extend(page_candidates)
            enriched_records.extend(page_enriched)
            accepted.extend(page_accepted)
            state["seen_candidates"] = len(all_entries)
            state["accepted_papers"] = len(accepted)
            state["pages_completed"] += 1
            state["last_page_start"] = start
            state["next_start"] = start + args.page_size
            print(
                json.dumps(
                    {
                        "event": "page_completed",
                        "page_start": start,
                        "page_seen": len(page_candidates),
                        "seen_candidates": len(all_entries),
                        "accepted_papers": len(accepted),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.no_sleep:
                sleep_between_requests(args.request_sleep)
            page_since_checkpoint += 1
            if (
                args.checkpoint_every_pages > 0
                and page_since_checkpoint >= args.checkpoint_every_pages
            ):
                _write_checkpoint(
                    candidates_path=candidates_path,
                    enriched_path=enriched_path,
                    accepted_path=accepted_path,
                    state_path=state_path,
                    new_candidates=page_candidates,
                    new_enriched=page_enriched,
                    new_accepted=page_accepted,
                    state=state,
                )
                page_since_checkpoint = 0
            if len(accepted) >= args.target_papers:
                break
    finally:
        progress.close()
        _write_checkpoint(
            candidates_path=candidates_path,
            enriched_path=enriched_path,
            accepted_path=accepted_path,
            state_path=state_path,
            new_candidates=[],
            new_enriched=[],
            new_accepted=[],
            state=state,
            final=True,
        )
        if close_client:
            client.close()

    return {
        "arxiv_candidates": len(all_entries),
        "openalex_enriched": len(enriched_records),
        "accepted_papers": len(accepted),
        "accepted_path": str(accepted_path),
        "state_path": str(state_path),
    }


def main() -> None:
    args = parse_args()
    summary = collect_stage2_papers(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["accepted_papers"] < args.target_papers:
        raise SystemExit(
            f"Accepted {summary['accepted_papers']} papers, below target {args.target_papers}. "
            "Increase --max-candidates or relax filters."
        )


if __name__ == "__main__":
    main()
