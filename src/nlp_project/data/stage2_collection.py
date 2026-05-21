from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ARXIV_API_URL = "https://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_ARXIV_SOURCE_ID = "S4306400194"
OPENALEX_COMPUTER_SCIENCE_FIELD_ID = "fields/17"
ARXIV_HTML_BASE_URL = "https://arxiv.org/html"
ARXIV_ABS_BASE_URL = "https://arxiv.org/abs"
ARXIV_PDF_BASE_URL = "https://arxiv.org/pdf"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_API_MAX_RETRIES = 5
ARXIV_API_INITIAL_BACKOFF_SECONDS = 5.0

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
    value = value.removeprefix("https://arxiv.org/pdf/")
    value = value.removeprefix("http://arxiv.org/pdf/")
    value = value.removeprefix("arXiv:")
    value = value.removesuffix(".pdf")
    return value.rstrip("/")


def extract_arxiv_id_from_url(value: str | None) -> str | None:
    if not value or "arxiv.org/" not in value:
        return None
    match = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^?#/]+)", value)
    if not match:
        return None
    return normalize_arxiv_id(match.group(1))


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


def _openalex_field_id(value: str | None) -> str:
    if not value:
        return ""
    return value.removeprefix("https://openalex.org/")


def is_openalex_cs_ai_work(record: dict[str, Any]) -> bool:
    primary_topic = record.get("primary_topic") or {}
    topics = [primary_topic, *(record.get("topics") or [])]
    for topic in topics:
        field = topic.get("field") or {}
        if _openalex_field_id(field.get("id")) == OPENALEX_COMPUTER_SCIENCE_FIELD_ID:
            return True
    return False


def extract_openalex_arxiv_id(record: dict[str, Any]) -> str | None:
    open_access = record.get("open_access") or {}
    arxiv_id = extract_arxiv_id_from_url(open_access.get("oa_url"))
    if arxiv_id:
        return arxiv_id
    for location in record.get("locations") or []:
        arxiv_id = extract_arxiv_id_from_url(location.get("landing_page_url"))
        if arxiv_id:
            return arxiv_id
        arxiv_id = extract_arxiv_id_from_url(location.get("pdf_url"))
        if arxiv_id:
            return arxiv_id
    ids = record.get("ids") or {}
    return normalize_arxiv_id(ids.get("arxiv"))


def _openalex_topic_categories(record: dict[str, Any]) -> list[str]:
    topics = record.get("topics") or []
    categories = []
    for topic in topics:
        name = _normalize_text(topic.get("display_name"))
        if name:
            categories.append(f"openalex:{name}")
    if categories:
        return categories
    primary_topic = record.get("primary_topic") or {}
    primary_name = _normalize_text(primary_topic.get("display_name"))
    return [f"openalex:{primary_name}"] if primary_name else ["openalex:computer_science"]


def make_openalex_paper_candidate(
    record: dict[str, Any],
    *,
    min_citations: int,
) -> PaperCandidate | None:
    cited_by_count = int(record.get("cited_by_count") or 0)
    if cited_by_count < min_citations:
        return None
    if not is_openalex_cs_ai_work(record):
        return None
    arxiv_id = extract_openalex_arxiv_id(record)
    if not arxiv_id:
        return None
    arxiv_base_id = strip_arxiv_version(arxiv_id)
    return PaperCandidate(
        arxiv_id=arxiv_id,
        arxiv_base_id=arxiv_base_id,
        title=_normalize_text(record.get("title") or record.get("display_name")),
        submitted_at=_normalize_text(record.get("publication_date")),
        categories=_openalex_topic_categories(record),
        cited_by_count=cited_by_count,
        html_url=f"{ARXIV_HTML_BASE_URL}/{arxiv_base_id}",
        pdf_url=f"{ARXIV_PDF_BASE_URL}/{arxiv_id}.pdf",
        source_url=f"{ARXIV_ABS_BASE_URL}/{arxiv_id}",
        license=(record.get("primary_location") or {}).get("license"),
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
                title=_normalize_text(_find_text(entry, "atom:title")),
                submitted_at=_normalize_text(_find_text(entry, "atom:published")),
                updated_at=_normalize_text(_find_text(entry, "atom:updated")) or None,
                summary=_normalize_text(_find_text(entry, "atom:summary")),
                categories=categories,
                pdf_url=pdf_url or f"{ARXIV_PDF_BASE_URL}/{arxiv_id}.pdf",
                source_url=f"{ARXIV_ABS_BASE_URL}/{arxiv_id}",
                html_url=f"{ARXIV_HTML_BASE_URL}/{strip_arxiv_version(arxiv_id)}",
                license=_normalize_text(_find_text(entry, "arxiv:license")) or None,
            )
        )
    return entries


def _find_text(element: ET.Element, path: str) -> str:
    found = element.find(path, ARXIV_NS)
    return found.text if found is not None and found.text else ""


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def build_arxiv_query(categories: Iterable[str]) -> str:
    return " OR ".join(f"cat:{category}" for category in categories)


def _retry_after_seconds(response: httpx.Response) -> float:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return 0.0
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return 0.0


def _should_retry_http_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    return isinstance(exc, httpx.TransportError)


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
        backoff = ARXIV_API_INITIAL_BACKOFF_SECONDS
        for attempt in range(1, ARXIV_API_MAX_RETRIES + 1):
            try:
                response = client.get(ARXIV_API_URL, params=params)
            except httpx.HTTPError as exc:
                if attempt >= ARXIV_API_MAX_RETRIES or not _should_retry_http_error(exc):
                    raise
                time.sleep(backoff)
                backoff *= 2
                continue
            if response.status_code == 200:
                return parse_arxiv_feed(response.text)
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
            if attempt >= ARXIV_API_MAX_RETRIES:
                response.raise_for_status()
            sleep_for = max(backoff, _retry_after_seconds(response))
            time.sleep(sleep_for)
            backoff *= 2
        raise RuntimeError("unreachable")
    finally:
        if close_client:
            client.close()


def fetch_openalex_arxiv_works_page(
    *,
    cursor: str = "*",
    per_page: int = 100,
    min_citations: int = 21,
    mailto: str | None = None,
    client: httpx.Client | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    filters = [
        f"locations.source.id:{OPENALEX_ARXIV_SOURCE_ID}",
        f"cited_by_count:>{min_citations - 1}",
        f"topics.field.id:{OPENALEX_COMPUTER_SCIENCE_FIELD_ID}",
    ]
    params = {
        "filter": ",".join(filters),
        "sort": "publication_date:desc",
        "per-page": str(per_page),
        "cursor": cursor,
    }
    if mailto:
        params["mailto"] = mailto
    close_client = client is None
    if client is None:
        client = httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        response = client.get(OPENALEX_WORKS_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        return payload.get("results") or [], (payload.get("meta") or {}).get("next_cursor")
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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
