from __future__ import annotations

import re
from dataclasses import asdict
from html import unescape
from typing import Any

from lxml import html as lxml_html

from nlp_project.data.processing import normalize_text
from nlp_project.data.stage2_collection import PaperCandidate

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
    re.compile(
        r"\[\s*"
        r"[A-Za-z][A-Za-z0-9:_./-]*\d+[A-Za-z0-9:_./-]*"
        r"(?:\s*[,;]\s*[A-Za-z][A-Za-z0-9:_./-]*\d+[A-Za-z0-9:_./-]*)+"
        r"\s*\]"
    ),
    re.compile(
        r"\((?:[A-Z][A-Za-z'`-]+(?:\s+et\s+al\.)?|\w+\s+and\s+\w+),?\s+\d{4}[a-z]?\)"
    ),
]

TOKEN_RE = re.compile(r"[A-Za-z]")
MATH_SEGMENT_RE = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+?\$)", re.DOTALL)
SECTION_CLASS_RE = re.compile(r"(?:^|\s)ltx_(?:sub)*section(?:\s|$)")


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


def collect_text_sections(root: Any) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    abstract_nodes = root.xpath(
        "//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_abstract ')]"
    )
    for node in abstract_nodes:
        paragraphs = [_paragraph_text(paragraph) for paragraph in _direct_section_paragraphs(node)]
        paragraphs = [
            paragraph
            for paragraph in paragraphs
            if is_useful_english_text(paragraph, min_chars=20) or contains_math_markup(paragraph)
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
        paragraphs = [_paragraph_text(paragraph) for paragraph in _direct_section_paragraphs(node)]
        paragraphs = [
            paragraph
            for paragraph in paragraphs
            if is_useful_english_text(paragraph) or contains_math_markup(paragraph)
        ]
        if paragraphs:
            sections.append(
                {"heading": heading or "Body", "kind": "body", "paragraphs": paragraphs}
            )
    return sections


def clean_plain_text(text: str) -> str:
    text = unescape(text)
    parts: list[str] = []
    cursor = 0
    for match in MATH_SEGMENT_RE.finditer(text):
        parts.append(_clean_non_math_text(text[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_clean_non_math_text(text[cursor:]))
    return "".join(parts).strip()


def contains_math_markup(text: str) -> bool:
    normalized = normalize_text(text)
    return "$$" in normalized or "$" in normalized


def is_useful_english_text(text: str, *, min_chars: int = 80) -> bool:
    text = normalize_text(text)
    if len(text) < min_chars:
        return False
    letters = TOKEN_RE.findall(text)
    return len(letters) / max(1, len(text)) >= 0.45


def _remove_unwanted_nodes(root: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    counts["math"] = _format_math_nodes(root)
    selectors = {
        "citations": (
            "//cite[contains(concat(' ', normalize-space(@class), ' '), ' ltx_cite ')]"
            "|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_cite ')]"
            "|//a[starts-with(@href, '#bib')]"
            "|//a[starts-with(@href, '#') and (contains(@href, '.T') or contains(@href, '.F'))]"
        ),
        "tables": (
            "//table[not(contains(concat(' ', normalize-space(@class), ' '), "
            "' ltx_equation ')) and not(contains(concat(' ', normalize-space(@class), ' '), "
            "' ltx_eqn_table '))]"
            "|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_table ')]"
            "|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_tabular ')]"
        ),
        "figures": (
            "//figure|//img|//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' ltx_figure ')]|//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' ltx_graphics ')]|//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' ltx_picture ')]"
        ),
        "captions": (
            "//figcaption|//*[contains(concat(' ', normalize-space(@class), ' '), "
            "' ltx_caption ')]"
        ),
        "references": (
            "//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_bibliography ')]"
            "|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_biblist ')]"
            "|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_bibitem ')]"
            "|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_thebibliography ')]"
            "|//*[@id='bib' or starts-with(@id, 'bib.')]"
        ),
        "scripts": "//script|//style|//noscript",
    }
    for name, xpath in selectors.items():
        nodes = root.xpath(xpath)
        counts[name] = len(nodes)
        for node in nodes:
            _drop_node_preserving_tail(node)
    return counts


def _format_math_nodes(root: Any) -> int:
    formatted = 0
    equation_containers = root.xpath(
        "//table[contains(concat(' ', normalize-space(@class), ' '), ' ltx_equation ')]"
        "|//table[contains(concat(' ', normalize-space(@class), ' '), ' ltx_eqn_table ')]"
        "|//div[contains(concat(' ', normalize-space(@class), ' '), ' ltx_equation ')]"
    )
    for container in equation_containers:
        latex_segments: list[str] = []
        for math_node in container.xpath(
            ".//math|.//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_Math ')]"
        ):
            latex = _math_node_to_latex(math_node)
            if latex:
                formatted += 1
                latex_segments.append(f"$${latex}$$")
        replacement_text = "\n\n".join(latex_segments).strip()
        if replacement_text:
            replacement = lxml_html.Element("p")
            replacement.text = replacement_text
            _replace_node(container, replacement)
        else:
            _drop_node_preserving_tail(container)

    math_nodes = root.xpath(
        "//math|//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_Math ')]"
    )
    for node in math_nodes:
        latex = _math_node_to_latex(node)
        if not latex:
            continue
        formatted += 1
        display = _math_display_mode(node)
        _replace_node_text(node, f"$${latex}$$" if display == "block" else f"${latex}$")
    return formatted


def _math_display_mode(node: Any) -> str:
    display = normalize_text(node.get("display")).casefold()
    if display in {"block", "display"}:
        return "block"
    parent = node.getparent()
    while parent is not None:
        classes = normalize_text(parent.get("class"))
        if "ltx_equation" in classes or "ltx_eqn_table" in classes:
            return "block"
        parent = parent.getparent()
    return "inline"


def _math_node_to_latex(node: Any) -> str:
    annotations = node.xpath(".//annotation[@encoding='application/x-tex']")
    for annotation in annotations:
        text = normalize_text("".join(annotation.itertext()))
        if text:
            return unescape(text)
    alttext = normalize_text(node.get("alttext"))
    if alttext:
        return unescape(alttext)
    return unescape(normalize_text("".join(node.itertext())))


def _replace_node(node: Any, replacement: Any) -> None:
    parent = node.getparent()
    if parent is None:
        return
    tail = node.tail
    node.tail = None
    node.addnext(replacement)
    replacement.tail = tail
    parent.remove(node)


def _replace_node_text(node: Any, text: str) -> None:
    for child in list(node):
        node.remove(child)
    node.text = text


def _drop_node_preserving_tail(node: Any) -> None:
    if hasattr(node, "drop_tree"):
        node.drop_tree()
        return
    parent = node.getparent()
    if parent is not None:
        parent.remove(node)


def _section_heading(node: Any) -> str:
    headings = node.xpath(
        ".//h1|.//h2|.//h3|.//h4|"
        ".//*[contains(concat(' ', normalize-space(@class), ' '), ' ltx_title ')]"
    )
    if not headings:
        return ""
    return clean_plain_text(" ".join(headings[0].itertext()))


def _is_drop_heading(heading: str) -> bool:
    normalized = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[A-Z])\.?\s+", "", heading.strip()).lower()
    if normalized in REFERENCE_HEADINGS:
        return True
    return any(normalized.startswith(prefix) for prefix in DROP_HEADING_PREFIXES)


def _direct_section_paragraphs(node: Any) -> list[Any]:
    return [
        paragraph
        for paragraph in node.xpath(".//p")
        if _nearest_section_like_ancestor(paragraph) is node
    ]


def _nearest_section_like_ancestor(node: Any) -> Any | None:
    parent = node.getparent()
    while parent is not None:
        if _is_section_like(parent):
            return parent
        parent = parent.getparent()
    return None


def _is_section_like(node: Any) -> bool:
    if getattr(node, "tag", None) == "section":
        return True
    classes = normalize_text(node.get("class"))
    return "ltx_abstract" in classes or bool(SECTION_CLASS_RE.search(classes))


def _paragraph_text(node: Any) -> str:
    return clean_plain_text(" ".join(node.itertext()))


def _clean_non_math_text(text: str) -> str:
    for pattern in INLINE_CITATION_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\b(?:Table|Figure|Fig\.)\s+\d+(?:\.\d+)*:?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([.,;:?!])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text
