from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from nlp_project.data.processing import normalize_text, split_by_stable_hash, write_jsonl
from nlp_project.data.stage2_collection import PaperCandidate, load_candidates
from nlp_project.data.stage2_html import build_stage2_document

MATH_SEGMENT_RE = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+?\$)", re.DOTALL)


def split_document_into_segments(
    document: dict[str, Any],
    *,
    min_chars: int = 180,
    target_chars: int = 1400,
    max_chars: int = 1800,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    paper_id = document["paper_id"]
    for section in document.get("sections", []):
        section_name = _normalize_section_name(
            section.get("heading") or section.get("kind") or "body"
        )
        buffer: list[str] = []
        buffer_len = 0
        for paragraph in section.get("paragraphs", []):
            paragraph = normalize_segment_text(paragraph)
            if not paragraph:
                continue
            if len(paragraph) > max_chars:
                if buffer:
                    _append_segment(segments, paper_id, section_name, buffer)
                    buffer = []
                    buffer_len = 0
                for chunk in split_long_text(paragraph, max_chars=max_chars):
                    if len(chunk) >= min_chars:
                        segments.append(
                            {"paper_id": paper_id, "section": section_name, "text": chunk}
                        )
                continue
            next_len = buffer_len + len(paragraph) + (2 if buffer else 0)
            if buffer and next_len > target_chars:
                _append_segment(segments, paper_id, section_name, buffer)
                buffer = [paragraph]
                buffer_len = len(paragraph)
            else:
                buffer.append(paragraph)
                buffer_len = next_len
        if buffer:
            text = "\n\n".join(buffer)
            if len(text) >= min_chars or section.get("kind") == "abstract":
                _append_segment(segments, paper_id, section_name, buffer)
    return segments


def _append_segment(
    segments: list[dict[str, Any]], paper_id: str, section_name: str, paragraphs: list[str]
) -> None:
    text = "\n\n".join(paragraphs).strip()
    if text:
        segments.append({"paper_id": paper_id, "section": section_name, "text": text})


def split_long_text(text: str, *, max_chars: int) -> list[str]:
    parts = split_text_preserving_math(text)
    if len(parts) > 1:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for part in parts:
            if contains_math_delimiters(part):
                if current:
                    chunks.extend(split_long_text(" ".join(current), max_chars=max_chars))
                    current = []
                    current_len = 0
                chunks.append(part.strip())
                continue
            for chunk in split_long_text(part, max_chars=max_chars):
                if current and current_len + len(chunk) + 1 > max_chars:
                    chunks.append(" ".join(current))
                    current = [chunk]
                    current_len = len(chunk)
                else:
                    current.append(chunk)
                    current_len += len(chunk) + 1
        if current:
            chunks.append(" ".join(current))
        return [chunk for chunk in chunks if chunk.strip()]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            chunks.extend(
                sentence[index : index + max_chars] for index in range(0, len(sentence), max_chars)
            )
            continue
        if current and current_len + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def normalize_segment_text(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for match in MATH_SEGMENT_RE.finditer(text):
        non_math = normalize_text(text[cursor : match.start()])
        if non_math:
            parts.append(non_math)
        math_text = match.group(0).strip()
        if math_text.startswith("$$"):
            parts.append(f"\n\n{math_text}\n\n")
        else:
            parts.append(math_text)
        cursor = match.end()
    tail = normalize_text(text[cursor:])
    if tail:
        parts.append(tail)
    normalized = " ".join(parts)
    normalized = re.sub(r"[ \t]*\n\n[ \t]*", "\n\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_text_preserving_math(text: str) -> list[str]:
    parts: list[str] = []
    cursor = 0
    for match in MATH_SEGMENT_RE.finditer(text):
        if match.start() > cursor:
            parts.append(text[cursor : match.start()].strip())
        parts.append(match.group(0).strip())
        cursor = match.end()
    if cursor < len(text):
        parts.append(text[cursor:].strip())
    return [part for part in parts if part]


def contains_math_delimiters(text: str) -> bool:
    return bool(MATH_SEGMENT_RE.fullmatch(text.strip()))


def _normalize_section_name(value: str) -> str:
    value = re.sub(r"^\d+(?:\.\d+)*\s*", "", value.strip().lower())
    if "abstract" in value:
        return "abstract"
    if "introduction" in value:
        return "introduction"
    if "conclusion" in value:
        return "conclusion"
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_") or "body"


def make_stage2_segment_record(
    *,
    paper_id: str,
    segment_id: str,
    section: str,
    text: str,
    title: str,
    source_url: str,
    cited_by_count: int,
    license: str | None,
    split: str,
) -> dict[str, Any]:
    return {
        "id": segment_id,
        "source": text,
        "domain": "cs_ai_paper",
        "split": split,
        "metadata": {
            "source_dataset": "arxiv",
            "paper_id": paper_id,
            "section": section,
            "title": title,
            "source_url": source_url,
            "cited_by_count": cited_by_count,
            "license": license,
        },
    }


def build_segments_for_document(
    candidate: PaperCandidate, document: dict[str, Any]
) -> list[dict[str, Any]]:
    split = split_by_stable_hash(candidate.arxiv_base_id, validation_pct=5, test_pct=5)
    raw_segments = split_document_into_segments(document)
    records = []
    for index, segment in enumerate(raw_segments, start=1):
        segment_id = f"{candidate.arxiv_id}_{segment['section']}_{index:04d}"
        records.append(
            make_stage2_segment_record(
                paper_id=candidate.arxiv_id,
                segment_id=segment_id,
                section=segment["section"],
                text=segment["text"],
                title=document.get("title") or candidate.title,
                source_url=candidate.html_url,
                cited_by_count=candidate.cited_by_count,
                license=candidate.license,
                split=split,
            )
        )
    return records


def write_stage2_segments(
    *,
    candidates_path: Path,
    html_dir: Path,
    documents_dir: Path,
    segments_path: Path,
    build_document: Callable[[PaperCandidate, str], dict[str, Any]] = build_stage2_document,
    show_progress: bool = False,
) -> dict[str, int | str]:
    candidates = load_candidates(candidates_path)
    all_segments: list[dict[str, Any]] = []
    papers_processed = 0
    papers_missing_html = 0
    progress = (
        tqdm(candidates, desc="Extracting Stage 2 HTML", unit="paper")
        if show_progress
        else candidates
    )
    for candidate in progress:
        html_path = html_dir / f"{candidate.arxiv_base_id}.html"
        if not html_path.is_file():
            papers_missing_html += 1
            if show_progress:
                progress.set_postfix(
                    processed=papers_processed,
                    missing=papers_missing_html,
                    segments=len(all_segments),
                )
            continue
        html_text = html_path.read_text(encoding="utf-8")
        document = build_document(candidate, html_text)
        write_json(documents_dir / f"{candidate.arxiv_id}.json", document)
        segments = build_segments_for_document(candidate, document)
        all_segments.extend(segments)
        papers_processed += 1
        if show_progress:
            progress.set_postfix(
                processed=papers_processed,
                missing=papers_missing_html,
                segments=len(all_segments),
            )
    segments_written = write_jsonl(segments_path, all_segments)
    return {
        "papers_total": len(candidates),
        "papers_processed": papers_processed,
        "papers_missing_html": papers_missing_html,
        "segments_written": segments_written,
        "segments_path": str(segments_path),
        "documents_dir": str(documents_dir),
    }


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
