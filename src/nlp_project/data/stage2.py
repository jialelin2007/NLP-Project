from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any

import httpx
from lxml import html as lxml_html

from nlp_project.data.processing import normalize_text, split_by_stable_hash, write_jsonl

ARXIV_API_URL = "https://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
ARXIV_HTML_BASE_URL = "https://arxiv.org/html"
ARXIV_ABS_BASE_URL = "https://arxiv.org/abs"
ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

DEFAULT_STAGE2_CATEGORIES = (
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.CV",
    "cs.IR",
    "cs.RO",
    "cs.SE",
    "cs.DS",
    "stat.ML",
)

REFERENCE_HEADINGS = {"references", "bibliography", "参考文献"}
DROP_HEADING_PREFIXES = (
    "acknowledgment",
    "acknowledgement",
    "appendix",
    "supplementary",
)

INLINE_CITATION_PATTERNS = [
    re.compile(r"\[(?:\s*\d+\s*(?:[-,;]\s*\d+\s*)*)\]"),
    re.compile(r"\[\s*[A-Za-z][A-Za-z0-9:_./-]*\d+[A-Za-z0-9:_./-]*\s*\]"),
    re.compile(r"\((?:[A-Z][A-Za-z'`-]+(?:\s+et\s+al\.)?|\w+\s+and\s+\w+),?\s+\d{4}[a-z]?\)"),
]

TOKEN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class PaperCandidate:
    arxiv_id: str
    arxiv_base_id: str
    title: str
    submitted_at: str
    categories: list[str]
    cited_by_count: int
    html_url: str
    pdf_url: str
    source_url: str
    license: str | None = None


@dataclass(frozen=True)
class OpenAlexWork:
    arxiv_id: str | None
    cited_by_count: int
    title: str | None
    doi: str | None
    openalex_id: str | None
    publication_date: str | None


@dataclass(frozen=True)
class ArxivEntry:
    arxiv_id: str
    arxiv_base_id: str
    title: str
    submitted_at: str
    updated_at: str | None
    summary: str
    categories: list[str]
    pdf_url: str
    source_url: str
    html_url: str
    license: str | None


def strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    value = value.removeprefix("https://arxiv.org/abs/")
    value = value.removeprefix("http://arxiv.org/abs/")
    value = value.removeprefix("arXiv:")
    return value.rstrip("/")


def parse_openalex_work(record: dict[str, Any]) -> OpenAlexWork:
    ids = record.get("ids") or {}
    return OpenAlexWork(
        arxiv_id=normalize_arxiv_id(ids.get("arxiv")),
        cited_by_count=int(record.get("cited_by_count") or 0),
        title=record.get("title") or record.get("display_name"),
        doi=record.get("doi"),
        openalex_id=record.get("id"),
        publication_date=record.get("publication_date"),
    )


def _normalize_search_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


def parse_arxiv_feed(xml_text: str) -> list[ArxivEntry]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id = normalize_arxiv_id(_find_text(entry, "atom:id")) or ""
        if not arxiv_id:
            continue
        categories = [
            category.attrib["term"]
            for category in entry.findall("atom:category", ARXIV_NS)
            if category.attrib.get("term")
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                arxiv_base_id=strip_arxiv_version(arxiv_id),
                title=normalize_text(_find_text(entry, "atom:title")),
                submitted_at=normalize_text(_find_text(entry, "atom:published")),
                updated_at=normalize_text(_find_text(entry, "atom:updated")) or None,
                summary=normalize_text(_find_text(entry, "atom:summary")),
                categories=categories,
                pdf_url=pdf_url or f"{ARXIV_PDF_BASE_URL}/{arxiv_id}.pdf",
                source_url=f"{ARXIV_ABS_BASE_URL}/{arxiv_id}",
                html_url=f"{ARXIV_HTML_BASE_URL}/{strip_arxiv_version(arxiv_id)}",
                license=normalize_text(_find_text(entry, "arxiv:license")) or None,
            )
        )
    return entries


def _find_text(element: ET.Element, path: str) -> str:
    found = element.find(path, ARXIV_NS)
    return found.text if found is not None and found.text else ""


def build_arxiv_query(categories: Iterable[str]) -> str:
    return " OR ".join(f"cat:{category}" for category in categories)


def fetch_arxiv_candidates(
    *,
    categories: Iterable[str] = DEFAULT_STAGE2_CATEGORIES,
    max_results: int = 100,
    start: int = 0,
    client: httpx.Client | None = None,
) -> list[ArxivEntry]:
    params = {
        "search_query": build_arxiv_query(categories),
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = client.get(ARXIV_API_URL, params=params)
        response.raise_for_status()
        return parse_arxiv_feed(response.text)
    finally:
        if close_client:
            client.close()


def deduplicate_latest_arxiv_entries(entries: Iterable[ArxivEntry]) -> list[ArxivEntry]:
    by_base_id: dict[str, ArxivEntry] = {}
    for entry in entries:
        current = by_base_id.get(entry.arxiv_base_id)
        if current is None or _arxiv_version_number(entry.arxiv_id) > _arxiv_version_number(
            current.arxiv_id
        ):
            by_base_id[entry.arxiv_base_id] = entry
    return sorted(by_base_id.values(), key=lambda item: item.submitted_at, reverse=True)


def _arxiv_version_number(arxiv_id: str) -> int:
    match = re.search(r"v(\d+)$", arxiv_id)
    return int(match.group(1)) if match else 0


def fetch_openalex_work(
    arxiv_base_id: str,
    *,
    title: str | None = None,
    client: httpx.Client | None = None,
    mailto: str | None = None,
) -> OpenAlexWork | None:
    params: dict[str, str] = {"search": title or arxiv_base_id}
    if mailto:
        params["mailto"] = mailto
    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = client.get(OPENALEX_WORKS_URL, params=params)
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        target_title = _normalize_search_text(title)
        if target_title:
            exact_matches = [
                result
                for result in results
                if _normalize_search_text(result.get("display_name")) == target_title
            ]
            if exact_matches:
                return parse_openalex_work(
                    max(exact_matches, key=lambda item: item.get("cited_by_count") or 0)
                )
        return parse_openalex_work(max(results, key=lambda item: item.get("cited_by_count") or 0))
    finally:
        if close_client:
            client.close()


def make_paper_candidate(entry: ArxivEntry, work: OpenAlexWork) -> PaperCandidate:
    return PaperCandidate(
        arxiv_id=entry.arxiv_id,
        arxiv_base_id=entry.arxiv_base_id,
        title=entry.title,
        submitted_at=entry.submitted_at,
        categories=entry.categories,
        cited_by_count=work.cited_by_count,
        html_url=entry.html_url,
        pdf_url=entry.pdf_url,
        source_url=entry.source_url,
        license=entry.license,
    )


def clean_arxiv_html(html_text: str, *, source_url: str) -> dict[str, Any]:
    root = lxml_html.fromstring(html_text)
    removed_counts = _remove_unwanted_nodes(root)
    title = extract_html_title(root)
    sections = collect_text_sections(root)
    return {
        "title": title,
        "source_url": source_url,
        "sections": sections,
        "removed_counts": removed_counts,
    }


def extract_html_title(root: Any) -> str:
    for selector in [
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_title ')]",
        "//h1",
        "//title",
    ]:
        values = [clean_plain_text(" ".join(node.itertext())) for node in root.xpath(selector)]
        values = [value for value in values if value]
        if values:
            return values[0]
    return ""


def _remove_unwanted_nodes(root: Any) -> dict[str, int]:
    selectors = {
        "tables": "//table",
        "figures": (
            "//figure|//img|//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' ltx_figure ')]"
        ),
        "references": (
            "//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_bibliography ')]"
            "|//*[contains(concat(' ', normalize-space(@class), ' '), 'ltx_biblist')]"
        ),
        "scripts": "//script|//style|//noscript",
    }
    counts: dict[str, int] = {}
    for name, xpath in selectors.items():
        nodes = root.xpath(xpath)
        counts[name] = len(nodes)
        for node in nodes:
            _drop_node_preserving_tail(node)
    math_nodes = root.xpath(
        "//math|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_Math ')]"
    )
    counts["math"] = len(math_nodes)
    for node in math_nodes:
        for annotation in node.xpath(".//annotation"):
            parent = annotation.getparent()
            if parent is not None:
                parent.remove(annotation)
    return counts


def _drop_node_preserving_tail(node: Any) -> None:
    if hasattr(node, "drop_tree"):
        node.drop_tree()
        return
    parent = node.getparent()
    if parent is not None:
        parent.remove(node)


def collect_text_sections(root: Any) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    abstract_nodes = root.xpath(
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_abstract ')]"
    )
    for node in abstract_nodes:
        paragraphs = [_paragraph_text(paragraph) for paragraph in node.xpath(".//p")]
        paragraphs = [
            paragraph for paragraph in paragraphs if is_useful_english_text(paragraph, min_chars=20)
        ]
        if paragraphs:
            sections.append({"heading": "Abstract", "kind": "abstract", "paragraphs": paragraphs})

    section_nodes = root.xpath(
        "//section|//*[contains(concat(' ', normalize-space(@class), ' '), "
        "' ltx_section ')]"
    )
    seen_nodes: set[int] = set()
    for node in section_nodes:
        node_id = id(node)
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        heading = _section_heading(node)
        if _is_drop_heading(heading):
            continue
        paragraphs = [_paragraph_text(paragraph) for paragraph in node.xpath(".//p")]
        paragraphs = [paragraph for paragraph in paragraphs if is_useful_english_text(paragraph)]
        if paragraphs:
            sections.append(
                {"heading": heading or "Body", "kind": "body", "paragraphs": paragraphs}
            )
    return sections


def _section_heading(node: Any) -> str:
    headings = node.xpath(
        ".//h1|.//h2|.//h3|.//h4|"
        ".//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_title ')]"
    )
    if not headings:
        return ""
    return clean_plain_text(" ".join(headings[0].itertext()))


def _is_drop_heading(heading: str) -> bool:
    normalized = heading.strip().lower()
    if normalized in REFERENCE_HEADINGS:
        return True
    return any(normalized.startswith(prefix) for prefix in DROP_HEADING_PREFIXES)


def _paragraph_text(node: Any) -> str:
    return clean_plain_text(" ".join(node.itertext()))


def clean_plain_text(text: str) -> str:
    text = unescape(text)
    for pattern in INLINE_CITATION_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_useful_english_text(text: str, *, min_chars: int = 80) -> bool:
    text = normalize_text(text)
    if len(text) < min_chars:
        return False
    letters = TOKEN_RE.findall(text)
    return len(letters) / max(1, len(text)) >= 0.45


def build_stage2_document(candidate: PaperCandidate, html_text: str) -> dict[str, Any]:
    document = clean_arxiv_html(html_text, source_url=candidate.html_url)
    document.update(
        {
            "paper_id": candidate.arxiv_id,
            "arxiv_base_id": candidate.arxiv_base_id,
            "title": document["title"] or candidate.title,
            "metadata": asdict(candidate),
        }
    )
    return document


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
            paragraph = normalize_text(paragraph)
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
                sentence[index : index + max_chars]
                for index in range(0, len(sentence), max_chars)
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


def process_html_files(
    *,
    candidates_path: Path,
    html_dir: Path,
    documents_dir: Path,
    segments_path: Path,
) -> dict[str, int | str]:
    candidates = load_candidates(candidates_path)
    all_segments: list[dict[str, Any]] = []
    papers_processed = 0
    papers_missing_html = 0
    for candidate in candidates:
        html_path = html_dir / f"{candidate.arxiv_base_id}.html"
        if not html_path.is_file():
            papers_missing_html += 1
            continue
        html_text = html_path.read_text(encoding="utf-8")
        document = build_stage2_document(candidate, html_text)
        write_json(documents_dir / f"{candidate.arxiv_id}.json", document)
        segments = build_segments_for_document(candidate, document)
        all_segments.extend(segments)
        papers_processed += 1
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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_candidates(path: Path, candidates: Iterable[PaperCandidate]) -> int:
    return write_jsonl(path, (asdict(candidate) for candidate in candidates))


def load_candidates(path: Path) -> list[PaperCandidate]:
    return [PaperCandidate(**record) for record in iter_jsonl(path)]


def fetch_text(url: str, *, client: httpx.Client | None = None) -> str:
    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        response = client.get(url)
        response.raise_for_status()
        return response.text
    finally:
        if close_client:
            client.close()


def has_usable_arxiv_html(candidate: PaperCandidate, *, client: httpx.Client | None = None) -> bool:
    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=20.0, follow_redirects=True)
    try:
        response = client.head(candidate.html_url)
        if response.status_code == 405:
            response = client.get(candidate.html_url)
        return response.status_code == 200 and "html" in response.headers.get("content-type", "")
    finally:
        if close_client:
            client.close()


def sleep_between_requests(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)
