from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import httpx
import pytest

from nlp_project.data.stage2_collection import (
    ArxivEntry,
    OpenAlexWork,
    make_openalex_paper_candidate,
)


def _load_collect_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts/data/stage2_collect_papers.py"
    spec = importlib.util.spec_from_file_location("stage2_collect_papers_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_fetch_html_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts/data/stage2_fetch_html.py"
    spec = importlib.util.spec_from_file_location("stage2_fetch_html_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _openalex_work(
    *,
    arxiv_url: str = "https://arxiv.org/abs/2401.12345v2",
    cited_by_count: int = 42,
    field_id: str = "https://openalex.org/fields/17",
) -> dict:
    return {
        "id": "https://openalex.org/W1",
        "title": "Paper title",
        "display_name": "Paper title",
        "publication_date": "2024-01-02",
        "cited_by_count": cited_by_count,
        "ids": {},
        "open_access": {"oa_url": arxiv_url},
        "locations": [],
        "primary_topic": {
            "display_name": "Artificial intelligence",
            "field": {"id": field_id, "display_name": "Computer science"},
        },
        "topics": [
            {
                "display_name": "Machine learning",
                "field": {"id": field_id, "display_name": "Computer science"},
            }
        ],
    }


def test_stage2_openalex_work_extracts_arxiv_id_and_candidate() -> None:
    record = _openalex_work(arxiv_url="https://arxiv.org/pdf/2401.12345v2.pdf")

    candidate = make_openalex_paper_candidate(record, min_citations=21)

    assert candidate is not None
    assert candidate.arxiv_id == "2401.12345v2"
    assert candidate.arxiv_base_id == "2401.12345"
    assert candidate.html_url == "https://arxiv.org/html/2401.12345"
    assert candidate.pdf_url == "https://arxiv.org/pdf/2401.12345v2.pdf"
    assert candidate.source_url == "https://arxiv.org/abs/2401.12345v2"
    assert candidate.categories == ["openalex:Machine learning"]


def test_stage2_openalex_work_filters_low_citation_and_non_cs() -> None:
    assert (
        make_openalex_paper_candidate(_openalex_work(cited_by_count=20), min_citations=21)
        is None
    )
    assert (
        make_openalex_paper_candidate(
            _openalex_work(field_id="https://openalex.org/fields/27"),
            min_citations=21,
        )
        is None
    )


def test_stage2_openalex_work_extracts_arxiv_id_from_locations_before_ids() -> None:
    record = _openalex_work(arxiv_url=None)
    record["locations"] = [
        {
            "landing_page_url": "https://example.org/not-arxiv",
            "pdf_url": "https://arxiv.org/pdf/2401.54321v1.pdf",
        }
    ]
    record["ids"] = {"arxiv": "https://arxiv.org/abs/2401.99999v1"}

    candidate = make_openalex_paper_candidate(record, min_citations=21)

    assert candidate is not None
    assert candidate.arxiv_id == "2401.54321v1"


def test_stage2_fetch_arxiv_candidates_retries_after_429(monkeypatch) -> None:
    module = _load_collect_script_module()
    xml_text = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom'>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-02T00:00:00Z</published>
    <title>Paper title</title>
    <summary>Abstract text.</summary>
    <author><name>Alice</name></author>
    <category term='cs.LG'/>
    <link title='pdf' href='https://arxiv.org/pdf/2401.12345v1.pdf' type='application/pdf'/>
    <arxiv:license>cc-by-4.0</arxiv:license>
  </entry>
</feed>
"""

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url, params=None):
            self.calls += 1
            request = httpx.Request("GET", url, params=params)
            if self.calls == 1:
                return httpx.Response(429, request=request, headers={"Retry-After": "0"})
            return httpx.Response(200, request=request, text=xml_text)

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        module.fetch_arxiv_candidates.__globals__["time"],
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    client = FakeClient()
    entries = module.fetch_arxiv_candidates(client=client)

    assert client.calls == 2
    assert sleep_calls
    assert entries and entries[0].arxiv_id == "2401.12345v1"


def test_stage2_fetch_arxiv_candidates_retries_after_timeout(monkeypatch) -> None:
    module = _load_collect_script_module()
    xml_text = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-02T00:00:00Z</published>
    <title>Paper title</title>
    <summary>Abstract text.</summary>
    <category term='cs.LG'/>
  </entry>
</feed>
"""

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url, params=None):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("simulated timeout")
            request = httpx.Request("GET", url, params=params)
            return httpx.Response(200, request=request, text=xml_text)

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        module.fetch_arxiv_candidates.__globals__["time"],
        "sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    client = FakeClient()
    entries = module.fetch_arxiv_candidates(client=client)

    assert client.calls == 2
    assert sleep_calls
    assert entries and entries[0].arxiv_id == "2401.12345v1"


def test_stage2_fetch_page_candidates_updates_progress_per_candidate(monkeypatch) -> None:
    module = _load_collect_script_module()
    entries = [
        ArxivEntry(
            arxiv_id="2401.12345v1",
            arxiv_base_id="2401.12345",
            title="Paper A",
            submitted_at="2024-01-02T00:00:00Z",
            updated_at=None,
            summary="Abstract A.",
            categories=["cs.LG"],
            pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
            source_url="https://arxiv.org/abs/2401.12345v1",
            html_url="https://arxiv.org/html/2401.12345",
            license="cc-by-4.0",
        ),
        ArxivEntry(
            arxiv_id="2401.12346v1",
            arxiv_base_id="2401.12346",
            title="Paper B",
            submitted_at="2024-01-01T00:00:00Z",
            updated_at=None,
            summary="Abstract B.",
            categories=["cs.LG"],
            pdf_url="https://arxiv.org/pdf/2401.12346v1.pdf",
            source_url="https://arxiv.org/abs/2401.12346v1",
            html_url="https://arxiv.org/html/2401.12346",
            license="cc-by-4.0",
        ),
    ]
    work = OpenAlexWork(
        arxiv_id="2401.12345v1",
        cited_by_count=42,
        title="Paper A",
        doi=None,
        openalex_id="https://openalex.org/W1",
        publication_date="2024-01-02",
    )
    updates: list[int] = []

    monkeypatch.setattr(
        module,
        "fetch_openalex_work",
        lambda arxiv_base_id, **kwargs: work,
    )
    monkeypatch.setattr(module, "has_usable_arxiv_html", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "sleep_between_requests", lambda seconds: None)

    page_candidates, page_enriched, page_accepted = module._fetch_page_candidates(
        entries,
        client=object(),
        mailto=None,
        openalex_workers=2,
        skip_html_check=True,
        html_check_client=object(),
        min_citations=21,
        no_sleep=True,
        existing_base_ids=set(),
        progress_callback=lambda n: updates.append(n),
    )

    assert len(page_candidates) == 2
    assert len(page_enriched) == 2
    assert len(page_accepted) == 2
    assert updates == [1, 1]


def test_stage2_collect_papers_checkpoints_progress_before_timeout(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _load_collect_script_module()
    raw_dir = tmp_path / "raw"
    entry = ArxivEntry(
        arxiv_id="2401.12345v1",
        arxiv_base_id="2401.12345",
        title="Paper title",
        submitted_at="2024-01-02T00:00:00Z",
        updated_at=None,
        summary="A useful abstract.",
        categories=["cs.LG"],
        pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
        source_url="https://arxiv.org/abs/2401.12345v1",
        html_url="https://arxiv.org/html/2401.12345",
        license="cc-by-4.0",
    )
    work = OpenAlexWork(
        arxiv_id="2401.12345v1",
        cited_by_count=42,
        title="Paper title",
        doi=None,
        openalex_id="https://openalex.org/W1",
        publication_date="2024-01-02",
    )

    def fake_fetch_arxiv_candidates(*, start: int, **kwargs):
        if start == 0:
            return [entry]
        raise httpx.ReadTimeout("simulated timeout")

    monkeypatch.setattr(module, "fetch_arxiv_candidates", fake_fetch_arxiv_candidates)
    monkeypatch.setattr(module, "fetch_openalex_work", lambda *args, **kwargs: work)
    monkeypatch.setattr(module, "has_usable_arxiv_html", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "sleep_between_requests", lambda seconds: None)

    args = module.parse_args(
        [
            "--raw-dir",
            str(raw_dir),
            "--discovery-source",
            "arxiv-api",
            "--target-papers",
            "2",
            "--max-candidates",
            "2",
            "--page-size",
            "1",
            "--checkpoint-every-records",
            "1",
            "--no-progress",
        ]
    )

    with pytest.raises(httpx.ReadTimeout):
        module.collect_stage2_papers(args, client=object())

    captured = capsys.readouterr()
    assert '"event": "checkpoint_written"' in captured.out
    assert '"accepted_papers": 1' in captured.out

    accepted_records = [
        json.loads(line) for line in (raw_dir / "accepted_papers.jsonl").read_text().splitlines()
    ]
    enriched_records = [
        json.loads(line) for line in (raw_dir / "openalex_enriched.jsonl").read_text().splitlines()
    ]
    candidate_records = [
        json.loads(line) for line in (raw_dir / "arxiv_candidates.jsonl").read_text().splitlines()
    ]
    assert len(accepted_records) == 1
    assert len(enriched_records) == 1
    assert len(candidate_records) == 1
    assert accepted_records[0]["arxiv_id"] == "2401.12345v1"


def test_stage2_collect_papers_resumes_from_existing_outputs(tmp_path: Path, monkeypatch) -> None:
    module = _load_collect_script_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    first_entry = ArxivEntry(
        arxiv_id="2401.12345v1",
        arxiv_base_id="2401.12345",
        title="Paper title",
        submitted_at="2024-01-02T00:00:00Z",
        updated_at=None,
        summary="A useful abstract.",
        categories=["cs.LG"],
        pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
        source_url="https://arxiv.org/abs/2401.12345v1",
        html_url="https://arxiv.org/html/2401.12345",
        license="cc-by-4.0",
    )
    second_entry = ArxivEntry(
        arxiv_id="2401.12346v1",
        arxiv_base_id="2401.12346",
        title="Another paper",
        submitted_at="2024-01-01T00:00:00Z",
        updated_at=None,
        summary="Another useful abstract.",
        categories=["cs.LG"],
        pdf_url="https://arxiv.org/pdf/2401.12346v1.pdf",
        source_url="https://arxiv.org/abs/2401.12346v1",
        html_url="https://arxiv.org/html/2401.12346",
        license="cc-by-4.0",
    )
    first_work = OpenAlexWork(
        arxiv_id="2401.12345v1",
        cited_by_count=42,
        title="Paper title",
        doi=None,
        openalex_id="https://openalex.org/W1",
        publication_date="2024-01-02",
    )
    second_work = OpenAlexWork(
        arxiv_id="2401.12346v1",
        cited_by_count=55,
        title="Another paper",
        doi=None,
        openalex_id="https://openalex.org/W2",
        publication_date="2024-01-01",
    )
    candidates_path = raw_dir / "arxiv_candidates.jsonl"
    enriched_path = raw_dir / "openalex_enriched.jsonl"
    accepted_path = raw_dir / "accepted_papers.jsonl"
    state_path = raw_dir / "collection_state.json"
    candidates_path.write_text(json.dumps(first_entry.__dict__) + "\n", encoding="utf-8")
    enriched_path.write_text(
        json.dumps({"arxiv": first_entry.__dict__, "openalex": first_work.__dict__}) + "\n",
        encoding="utf-8",
    )
    accepted_path.write_text(
        json.dumps(
            {
                "arxiv_id": "2401.12345v1",
                "arxiv_base_id": "2401.12345",
                "title": "Paper title",
                "submitted_at": "2024-01-02T00:00:00Z",
                "categories": ["cs.LG"],
                "cited_by_count": 42,
                "html_url": "https://arxiv.org/html/2401.12345",
                "pdf_url": "https://arxiv.org/pdf/2401.12345v1.pdf",
                "source_url": "https://arxiv.org/abs/2401.12345v1",
                "license": "cc-by-4.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "next_start": 1,
                "pages_completed": 1,
                "last_page_start": 0,
                "accepted_papers": 1,
                "seen_candidates": 1,
            }
        ),
        encoding="utf-8",
    )

    calls: list[int] = []

    def fake_fetch_arxiv_candidates(*, start: int, **kwargs):
        calls.append(start)
        if start == 1:
            return [second_entry]
        raise AssertionError(f"unexpected page start: {start}")

    monkeypatch.setattr(module, "fetch_arxiv_candidates", fake_fetch_arxiv_candidates)
    monkeypatch.setattr(
        module,
        "fetch_openalex_work",
        lambda arxiv_base_id, **kwargs: (
            second_work if arxiv_base_id == second_entry.arxiv_base_id else first_work
        ),
    )
    monkeypatch.setattr(module, "has_usable_arxiv_html", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "sleep_between_requests", lambda seconds: None)

    args = module.parse_args(
        [
            "--raw-dir",
            str(raw_dir),
            "--discovery-source",
            "arxiv-api",
            "--target-papers",
            "2",
            "--max-candidates",
            "3",
            "--page-size",
            "1",
            "--checkpoint-every-pages",
            "1",
            "--no-progress",
        ]
    )

    summary = module.collect_stage2_papers(args, client=object())

    assert summary["accepted_papers"] == 2
    assert calls == [1]
    assert candidates_path.read_text(encoding="utf-8").count("\n") == 2
    assert accepted_path.read_text(encoding="utf-8").count("\n") == 2


def test_stage2_collect_papers_openalex_first_writes_accepted(tmp_path: Path, monkeypatch) -> None:
    module = _load_collect_script_module()
    raw_dir = tmp_path / "raw"
    records = [
        _openalex_work(arxiv_url="https://arxiv.org/abs/2401.12345v1", cited_by_count=42),
        _openalex_work(arxiv_url="https://arxiv.org/abs/2401.12345v2", cited_by_count=50),
        _openalex_work(arxiv_url="https://arxiv.org/abs/2401.12346v1", cited_by_count=55),
    ]

    calls: list[str] = []

    def fake_fetch_openalex_page(*, cursor: str, **kwargs):
        calls.append(cursor)
        if cursor == "*":
            return [records[0], records[1]], "cursor-2"
        if cursor == "cursor-2":
            return [records[2]], None
        raise AssertionError(f"unexpected cursor {cursor}")

    monkeypatch.setattr(module, "fetch_openalex_arxiv_works_page", fake_fetch_openalex_page)
    monkeypatch.setattr(module, "has_usable_arxiv_html", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "sleep_between_requests", lambda seconds: None)

    args = module.parse_args(
        [
            "--raw-dir",
            str(raw_dir),
            "--target-papers",
            "2",
            "--page-size",
            "2",
            "--no-progress",
        ]
    )

    summary = module.collect_stage2_papers(args, client=object())

    accepted_records = [
        json.loads(line) for line in (raw_dir / "accepted_papers.jsonl").read_text().splitlines()
    ]
    enriched_records = [
        json.loads(line) for line in (raw_dir / "openalex_enriched.jsonl").read_text().splitlines()
    ]
    state = json.loads((raw_dir / "collection_state.json").read_text(encoding="utf-8"))
    assert summary["accepted_papers"] == 2
    assert calls == ["*", "cursor-2"]
    assert [record["arxiv_base_id"] for record in accepted_records] == [
        "2401.12345",
        "2401.12346",
    ]
    assert len(enriched_records) == 3
    assert state["openalex_cursor"] is None


def test_stage2_collect_papers_openalex_first_resumes_cursor(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_collect_script_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    accepted_path = raw_dir / "accepted_papers.jsonl"
    enriched_path = raw_dir / "openalex_enriched.jsonl"
    state_path = raw_dir / "collection_state.json"
    accepted_path.write_text(
        json.dumps(
            {
                "arxiv_id": "2401.12345v1",
                "arxiv_base_id": "2401.12345",
                "title": "Paper title",
                "submitted_at": "2024-01-02",
                "categories": ["openalex:Machine learning"],
                "cited_by_count": 42,
                "html_url": "https://arxiv.org/html/2401.12345",
                "pdf_url": "https://arxiv.org/pdf/2401.12345v1.pdf",
                "source_url": "https://arxiv.org/abs/2401.12345v1",
                "license": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    enriched_path.write_text(json.dumps({"openalex": _openalex_work()}) + "\n", encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "openalex_cursor": "cursor-2",
                "pages_completed": 1,
                "accepted_papers": 1,
                "seen_candidates": 1,
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_fetch_openalex_page(*, cursor: str, **kwargs):
        calls.append(cursor)
        return [_openalex_work(arxiv_url="https://arxiv.org/abs/2401.12346v1")], None

    monkeypatch.setattr(module, "fetch_openalex_arxiv_works_page", fake_fetch_openalex_page)
    monkeypatch.setattr(module, "has_usable_arxiv_html", lambda *args, **kwargs: True)
    monkeypatch.setattr(module, "sleep_between_requests", lambda seconds: None)

    args = module.parse_args(
        [
            "--raw-dir",
            str(raw_dir),
            "--target-papers",
            "2",
            "--page-size",
            "2",
            "--no-progress",
        ]
    )

    summary = module.collect_stage2_papers(args, client=object())

    assert summary["accepted_papers"] == 2
    assert calls == ["cursor-2"]
    assert accepted_path.read_text(encoding="utf-8").count("\n") == 2


def test_stage2_fetch_html_uses_concurrency(tmp_path: Path, monkeypatch) -> None:
    module = _load_fetch_html_script_module()
    raw_dir = tmp_path / "raw"
    html_dir = raw_dir / "html"
    html_dir.mkdir(parents=True)
    accepted_path = raw_dir / "accepted_papers.jsonl"
    accepted_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "arxiv_id": "2401.12345v1",
                        "arxiv_base_id": "2401.12345",
                        "title": "Paper A",
                        "submitted_at": "2024-01-02",
                        "categories": ["cs.LG"],
                        "cited_by_count": 42,
                        "html_url": "https://arxiv.org/html/2401.12345",
                        "pdf_url": "https://arxiv.org/pdf/2401.12345v1.pdf",
                        "source_url": "https://arxiv.org/abs/2401.12345v1",
                        "license": "cc-by-4.0",
                    }
                ),
                json.dumps(
                    {
                        "arxiv_id": "2401.12346v1",
                        "arxiv_base_id": "2401.12346",
                        "title": "Paper B",
                        "submitted_at": "2024-01-01",
                        "categories": ["cs.LG"],
                        "cited_by_count": 43,
                        "html_url": "https://arxiv.org/html/2401.12346",
                        "pdf_url": "https://arxiv.org/pdf/2401.12346v1.pdf",
                        "source_url": "https://arxiv.org/abs/2401.12346v1",
                        "license": "cc-by-4.0",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    active = 0
    peak_active = 0

    monkeypatch.setattr(
        module,
        "load_candidates",
        lambda path: [
            type(
                "Candidate",
                (),
                {
                    "arxiv_id": "2401.12345v1",
                    "arxiv_base_id": "2401.12345",
                    "html_url": "https://arxiv.org/html/2401.12345",
                },
            )(),
            type(
                "Candidate",
                (),
                {
                    "arxiv_id": "2401.12346v1",
                    "arxiv_base_id": "2401.12346",
                    "html_url": "https://arxiv.org/html/2401.12346",
                },
            )(),
        ],
    )

    def fake_fetch_text(url, client=None):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        time.sleep(0.2)
        active -= 1
        return "<html></html>"

    monkeypatch.setattr(module, "fetch_text", fake_fetch_text)
    monkeypatch.setattr(module, "sleep_between_requests", lambda seconds: None)

    args = module.parse_args(
        [
            "--accepted-papers",
            str(accepted_path),
            "--html-dir",
            str(html_dir),
            "--max-workers",
            "2",
            "--request-sleep",
            "0",
        ]
    )

    summary = module.fetch_html_files(args)
    assert summary["fetched"] == 2
    assert peak_active >= 2


def test_stage2_fetch_html_records_failures_without_exiting(tmp_path: Path, monkeypatch) -> None:
    module = _load_fetch_html_script_module()
    raw_dir = tmp_path / "raw"
    html_dir = raw_dir / "html"
    html_dir.mkdir(parents=True)
    accepted_path = raw_dir / "accepted_papers.jsonl"
    accepted_path.write_text(
        json.dumps(
            {
                "arxiv_id": "2401.12345v1",
                "arxiv_base_id": "2401.12345",
                "title": "Paper A",
                "submitted_at": "2024-01-02",
                "categories": ["cs.LG"],
                "cited_by_count": 42,
                "html_url": "https://arxiv.org/html/2401.12345",
                "pdf_url": "https://arxiv.org/pdf/2401.12345v1.pdf",
                "source_url": "https://arxiv.org/abs/2401.12345v1",
                "license": "cc-by-4.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "fetch_text",
        lambda url, client=None: (_ for _ in ()).throw(
            httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("GET", url),
                response=httpx.Response(404),
            )
        ),
    )
    monkeypatch.setattr(module, "sleep_between_requests", lambda seconds: None)

    args = module.parse_args(
        [
            "--accepted-papers",
            str(accepted_path),
            "--html-dir",
            str(html_dir),
            "--max-workers",
            "1",
            "--request-sleep",
            "0",
        ]
    )

    summary = module.fetch_html_files(args)

    assert summary["fetched"] == 0
    assert len(summary["failed"]) == 1
