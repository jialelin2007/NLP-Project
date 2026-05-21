from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_stage2_extract_segments_script_writes_all_jsonl(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    html_dir = raw_dir / "html"
    processed_dir = tmp_path / "processed"
    html_dir.mkdir(parents=True)
    accepted_path = raw_dir / "accepted_papers.jsonl"
    accepted_path.write_text(
        json.dumps(
            {
                "arxiv_id": "2401.12345v1",
                "arxiv_base_id": "2401.12345",
                "title": "Paper title",
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
    (html_dir / "2401.12345.html").write_text(
        """
        <html><body>
        <h1>Paper title</h1>
        <div class="ltx_abstract"><p>This abstract contains enough English text to survive
        the extraction threshold and become a useful training segment for translation.</p></div>
        <section><h2>Introduction</h2><p>This body paragraph contains enough English text
        to become another useful training segment for the downstream paper translation data
        preparation pipeline. It adds enough context about models, datasets, evaluation,
        and methods so the segment passes the minimum character threshold.</p></section>
        </body></html>
        """,
        encoding="utf-8",
    )
    segments_path = processed_dir / "segments" / "all.jsonl"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/data/stage2_extract_segments.py",
            "--accepted-papers",
            str(accepted_path),
            "--html-dir",
            str(html_dir),
            "--documents-dir",
            str(processed_dir / "documents"),
            "--segments-path",
            str(segments_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"segments_written": 2' in result.stdout
    assert segments_path.is_file()
    records = [json.loads(line) for line in segments_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
