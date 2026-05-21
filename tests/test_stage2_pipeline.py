from __future__ import annotations

import json
from pathlib import Path

from nlp_project.data.stage2 import (
    PaperCandidate,
    build_segments_for_document,
    clean_arxiv_html,
    make_stage2_segment_record,
    parse_openalex_work,
    process_html_files,
    split_document_into_segments,
)


def test_parse_openalex_work_reads_citation_count_and_arxiv_id() -> None:
    record = {
        "id": "https://openalex.org/W123",
        "cited_by_count": 42,
        "ids": {"arxiv": "https://arxiv.org/abs/2401.12345v2"},
        "publication_date": "2024-01-02",
        "title": "Paper title",
        "doi": "https://doi.org/10.1234/example",
        "primary_location": {"source": {"host_organization_name": "arXiv"}},
    }

    work = parse_openalex_work(record)

    assert work.arxiv_id == "2401.12345v2"
    assert work.cited_by_count == 42
    assert work.title == "Paper title"


def test_clean_arxiv_html_removes_tables_figures_references_math_and_inline_citations() -> None:
    html = """
    <html>
      <body>
        <h1>Title</h1>
        <div class="ltx_abstract">
          <p>Abstract with [12] citation and <math>x+y</math>.</p>
        </div>
        <section>
          <h2>Introduction</h2>
          <p>First paragraph (Smith et al., 2020) with <span class="ltx_Math">a=b</span>.</p>
          <figure><img src="x.png" alt="img"><figcaption>Figure 1</figcaption></figure>
          <table><tr><td>cell</td></tr></table>
          <div class="ltx_bibliography">References</div>
        </section>
      </body>
    </html>
    """

    doc = clean_arxiv_html(html, source_url="https://arxiv.org/html/2401.12345")

    assert "cell" not in json.dumps(doc, ensure_ascii=False)
    assert "Figure 1" not in json.dumps(doc, ensure_ascii=False)
    assert "[12]" not in json.dumps(doc, ensure_ascii=False)
    assert "Smith et al., 2020" not in json.dumps(doc, ensure_ascii=False)
    assert "x+y" not in json.dumps(doc, ensure_ascii=False)
    assert doc["title"] == "Title"
    assert doc["sections"][0]["heading"] == "Abstract"
    assert doc["sections"][0]["paragraphs"][0] == "Abstract with citation and."


def test_split_document_into_segments_preserves_abstract_and_stays_within_bounds() -> None:
    document = {
        "paper_id": "2401.12345v1",
        "title": "Paper title",
        "source_url": "https://arxiv.org/html/2401.12345",
        "sections": [
            {"heading": "Abstract", "kind": "abstract", "paragraphs": ["A" * 220]},
            {"heading": "Introduction", "kind": "body", "paragraphs": ["B" * 500, "C" * 700]},
        ],
    }

    segments = split_document_into_segments(document)

    assert segments[0]["section"] == "abstract"
    assert segments[0]["text"] == "A" * 220
    assert all(len(segment["text"]) <= 1800 for segment in segments)
    assert {segment["paper_id"] for segment in segments} == {"2401.12345v1"}


def test_build_segments_for_document_assigns_metadata_and_split() -> None:
    candidate = PaperCandidate(
        arxiv_id="2401.12345v1",
        arxiv_base_id="2401.12345",
        title="Paper title",
        submitted_at="2024-01-02",
        categories=["cs.LG"],
        cited_by_count=42,
        html_url="https://arxiv.org/html/2401.12345",
        pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
        source_url="https://arxiv.org/abs/2401.12345v1",
        license="cc-by-4.0",
    )
    document = {
        "paper_id": "2401.12345v1",
        "title": "Paper title",
        "source_url": candidate.html_url,
        "sections": [
            {"heading": "Abstract", "kind": "abstract", "paragraphs": ["A" * 220]},
            {"heading": "Introduction", "kind": "body", "paragraphs": ["B" * 500]},
        ],
    }

    records = build_segments_for_document(candidate, document)

    assert records
    assert records[0]["metadata"]["cited_by_count"] == 42
    assert records[0]["metadata"]["source_dataset"] == "arxiv"
    assert records[0]["split"] in {"train", "validation", "test"}
    assert records[0]["domain"] == "cs_ai_paper"


def test_make_stage2_segment_record_uses_expected_schema() -> None:
    record = make_stage2_segment_record(
        paper_id="2401.12345v1",
        segment_id="2401.12345v1_abstract_0001",
        section="abstract",
        text="Some English text.",
        title="Paper title",
        source_url="https://arxiv.org/html/2401.12345",
        cited_by_count=42,
        license="cc-by-4.0",
        split="train",
    )

    assert record["id"] == "2401.12345v1_abstract_0001"
    assert record["split"] == "train"
    assert record["source"] == "Some English text."
    assert record["metadata"]["paper_id"] == "2401.12345v1"
    assert record["metadata"]["section"] == "abstract"
    assert record["metadata"]["cited_by_count"] == 42


def test_process_html_files_writes_documents_and_segments(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    html_dir = raw_dir / "html"
    html_dir.mkdir(parents=True)
    candidate = PaperCandidate(
        arxiv_id="2401.12345v1",
        arxiv_base_id="2401.12345",
        title="Paper title",
        submitted_at="2024-01-02",
        categories=["cs.LG"],
        cited_by_count=42,
        html_url="https://arxiv.org/html/2401.12345",
        pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
        source_url="https://arxiv.org/abs/2401.12345v1",
        license="cc-by-4.0",
    )
    candidates_path = raw_dir / "accepted_papers.jsonl"
    candidates_path.write_text(json.dumps(candidate.__dict__) + "\n", encoding="utf-8")
    (html_dir / "2401.12345.html").write_text(
        """
        <html><body>
        <h1>Paper title</h1>
        <div class="ltx_abstract"><p>This abstract contains enough English text to survive
        the extraction threshold and become a useful training segment for translation.</p></div>
        <section><h2>Introduction</h2><p>This body paragraph contains enough English text
        to become another useful training segment for the downstream paper translation data
        preparation pipeline. It adds a second sentence with enough context about models,
        datasets, evaluation, and methods so the segment passes the minimum character
        threshold used by the Stage 2 extraction pipeline.</p></section>
        </body></html>
        """,
        encoding="utf-8",
    )

    summary = process_html_files(
        candidates_path=candidates_path,
        html_dir=html_dir,
        documents_dir=processed_dir / "documents",
        segments_path=processed_dir / "segments" / "all.jsonl",
    )

    assert summary["papers_processed"] == 1
    assert summary["segments_written"] == 2
    assert (processed_dir / "documents" / "2401.12345v1.json").is_file()
    segments_text = (processed_dir / "segments" / "all.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in segments_text.splitlines()]
    assert {record["metadata"]["section"] for record in records} == {"abstract", "introduction"}
