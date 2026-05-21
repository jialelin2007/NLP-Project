from __future__ import annotations

import json

from nlp_project.data.stage2_collection import PaperCandidate
from nlp_project.data.stage2_html import (
    build_stage2_document,
    clean_arxiv_html,
    clean_plain_text,
    contains_math_markup,
)


def test_clean_arxiv_html_keeps_inline_math_and_removes_noise() -> None:
    html = """
    <html>
      <body>
        <h1>Title</h1>
        <div class="ltx_abstract">
          <p>Abstract with [12] citation and <math>x+y</math>.</p>
        </div>
        <section>
          <h2>Introduction</h2>
          <p>First paragraph (Smith et al., 2020) with <span class="ltx_Math">a=b</span> in the
          displayed equation context. This additional sentence keeps the paragraph long
          enough for Stage 2 extraction while preserving the inline formula for translation.</p>
          <p>Inline math with TeX source <math class="ltx_Math" display="inline" alttext="q(t)">
            <semantics><mi>q</mi><mo>(</mo><mi>t</mi><mo>)</mo>
              <annotation encoding="application/x-tex">q(t)</annotation>
            </semantics>
          </math>
          should be preserved as inline latex.</p>
          <div class="ltx_equation ltx_eqn_table">
            <table>
              <tbody>
                <tr>
                  <td class="ltx_eqn_cell ltx_align_center">
                  <math class="ltx_Math" display="block" alttext="\\mathcal{L}=x^2">
                    <semantics><mrow><mi>ℒ</mi><mo>=</mo><msup><mi>x</mi><mn>2</mn></msup></mrow>
                      <annotation encoding="application/x-tex">\\mathcal{L}=x^2</annotation>
                    </semantics>
                  </math></td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>follow MOTIVE [ wu2026motion ] for video generation and motion transfer
          experiments, where the baseline remains useful for comparing temporal
          coherence, prompt alignment, and controllable scene dynamics.</p>
          <p>Laten [ sohl2015deep, ho2020denoising, song2019generative, song2020score ].
          This sentence is intentionally long enough to survive extraction and should keep
          the surrounding words while removing the bracketed citation list.</p>
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
    assert "wu2026motion" not in json.dumps(doc, ensure_ascii=False)
    assert "sohl2015deep" not in json.dumps(doc, ensure_ascii=False)
    assert "$q(t)$" in json.dumps(doc, ensure_ascii=False)
    assert "$$\\\\mathcal{L}=x^2$$" in json.dumps(doc, ensure_ascii=False)
    assert doc["title"] == "Title"
    assert doc["sections"][0]["heading"] == "Abstract"
    assert doc["sections"][0]["paragraphs"][0] == "Abstract with citation and $x+y$."
    assert "follow MOTIVE for video generation" in json.dumps(doc, ensure_ascii=False)
    assert "Laten" in json.dumps(doc, ensure_ascii=False)


def test_clean_arxiv_html_removes_dom_citations_and_numbered_references() -> None:
    html = """
    <html><body>
      <h1>Title</h1>
      <section>
        <h2>Introduction</h2>
        <p>This paragraph cites
        <cite class="ltx_cite">Cabalar (<a href="#bib.bib5" class="ltx_ref">2011</a>);
        Lifschitz (<a href="#bib.bib10" class="ltx_ref">2012</a>)</cite>
        but should keep enough ordinary English words for extraction and preserve
        non-bibliographic references such as Section <a href="#S2" class="ltx_ref">2</a>
        and Equation <a href="#S2.E1" class="ltx_ref">(1)</a>.</p>
        <p>Bracket keys [Smith2020, Jones2021] and numeric ranges [1, 2, 3] are removed
        while the rest of this sufficiently long paragraph remains useful training text.</p>
      </section>
      <section id="bib" class="ltx_bibliography">
        <h2 class="ltx_title">7 References</h2>
        <ul class="ltx_biblist"><li class="ltx_bibitem">Cabalar 2011 bibliography text</li></ul>
      </section>
    </body></html>
    """

    doc = clean_arxiv_html(html, source_url="https://arxiv.org/html/2401.12345")
    text = json.dumps(doc, ensure_ascii=False)

    assert "Cabalar" not in text
    assert "Lifschitz" not in text
    assert "Smith2020" not in text
    assert "Jones2021" not in text
    assert "[1, 2, 3]" not in text
    assert "bibliography text" not in text
    assert "References" not in text
    assert "Section 2" in text
    assert "Equation (1)" in text


def test_clean_arxiv_html_removes_figures_captions_and_plain_tables_but_keeps_equations() -> None:
    html = r"""
    <html><body>
      <h1>Title</h1>
      <section>
        <h2>Method</h2>
        <p>This paragraph is long enough to remain after cleaning and it introduces
        the displayed formula below with useful context for translation data. The results
        shown in Table <a href="#S4.T1" class="ltx_ref">1</a> are not retained as a
        table reference in the cleaned training text.</p>
        <figure class="ltx_figure">
          <img class="ltx_graphics" src="figure.png" alt="Figure image">
          <figcaption class="ltx_caption">Figure 1: remove this caption</figcaption>
        </figure>
        <table class="ltx_table"><tr><td>Table 1 plain table cell text</td></tr></table>
        <div class="ltx_equation ltx_eqn_table">
          <table><tr><td><math display="block" alttext="\alpha+\beta">
            <annotation encoding="application/x-tex">\alpha+\beta</annotation>
          </math></td></tr></table>
        </div>
      </section>
    </body></html>
    """

    doc = clean_arxiv_html(html, source_url="https://arxiv.org/html/2401.12345")
    text = json.dumps(doc, ensure_ascii=False)

    assert "Figure 1" not in text
    assert "Figure image" not in text
    assert "Table 1 plain table cell text" not in text
    assert "Table 1" not in text
    assert "$$\\\\alpha+\\\\beta$$" in text


def test_clean_arxiv_html_does_not_duplicate_nested_section_paragraphs() -> None:
    html = """
    <html><body>
      <h1>Title</h1>
      <section>
        <h2>Outer</h2>
        <p>Outer paragraph has enough English text to survive the extraction threshold
        without needing content from the nested section below.</p>
        <section>
          <h3>Inner</h3>
          <p>Inner paragraph has enough English text to survive the extraction threshold
          and should appear only in the inner section output.</p>
        </section>
      </section>
    </body></html>
    """

    doc = clean_arxiv_html(html, source_url="https://arxiv.org/html/2401.12345")

    outer = [section for section in doc["sections"] if section["heading"] == "Outer"][0]
    inner = [section for section in doc["sections"] if section["heading"] == "Inner"][0]
    assert len(outer["paragraphs"]) == 1
    assert "Inner paragraph" not in outer["paragraphs"][0]
    assert len(inner["paragraphs"]) == 1


def test_clean_plain_text_preserves_math_and_strips_citations() -> None:
    text = "Alpha [12] beta $x+y$ and $$\\mathcal{L}=x^2$$."

    cleaned = clean_plain_text(text)

    assert cleaned == "Alpha beta $x+y$ and $$\\mathcal{L}=x^2$$."


def test_contains_math_markup_detects_math_delimiters() -> None:
    assert contains_math_markup("Text $x$") is True
    assert contains_math_markup("Text $$x$$") is True
    assert contains_math_markup("Text only") is False


def test_build_stage2_document_uses_candidate_metadata() -> None:
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

    document = build_stage2_document(
        candidate,
        """
        <html><body>
        <h1>Paper title</h1>
        <div class="ltx_abstract"><p>This abstract contains enough English text to survive
        the extraction threshold and become a useful training segment for translation.</p></div>
        </body></html>
        """,
    )

    assert document["paper_id"] == "2401.12345v1"
    assert document["metadata"]["cited_by_count"] == 42
    assert document["sections"][0]["heading"] == "Abstract"
