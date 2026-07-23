from pathlib import Path

from modules.ui.contracts import CorpusSnapshot, EvaluationSummary, RuntimeSnapshot
from modules.ui.presenters import (
    render_evidence,
    render_latest_evaluation,
    render_result_table,
    render_sidebar_status,
    render_status,
)

UI_DIR = Path(__file__).parents[2] / "modules" / "ui"


def test_status_presenter_escapes_dynamic_content_and_sets_live_semantics() -> None:
    # Act
    html = render_status("error", "<script>bad()</script>", "Try <again>.")

    # Assert status content is safe and announced by assistive technology.
    assert "<script>" not in html
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert "Try &lt;again&gt;." in html
    assert 'role="alert"' in html
    assert 'aria-live="assertive"' in html


def test_table_presenter_escapes_headers_rows_and_empty_messages() -> None:
    # Act
    html = render_result_table(
        ["<Document>"],
        [["<manual.txt>"]],
        caption="<Inventory>",
        empty_message="<Empty>",
    )
    empty = render_result_table(
        ["Document"],
        [],
        caption="Inventory",
        empty_message="<No documents>",
    )

    # Assert every dynamic table value is escaped.
    assert "<manual.txt>" not in html
    assert "&lt;manual.txt&gt;" in html
    assert "&lt;Document&gt;" in html
    assert "&lt;Inventory&gt;" in html
    assert "&lt;No documents&gt;" in empty


def test_evidence_presenter_exposes_only_escaped_public_source_fields() -> None:
    # Act
    html = render_evidence(
        [
            {
                "label": "C1",
                "filename": "<manual.pdf>",
                "page": 2,
                "excerpt": "<private-looking text>",
                "relevant": True,
                "prompt": "must not appear",
                "reasoning": "must not appear either",
            }
        ]
    )

    # Assert private graph fields never enter rendered evidence.
    assert "&lt;manual.pdf&gt;" in html
    assert "&lt;private-looking text&gt;" in html
    assert "must not appear" not in html
    assert "must not appear either" not in html


def test_sidebar_status_renders_only_models_and_complete_corpus_state() -> None:
    # Arrange
    runtime = RuntimeSnapshot(
        state="blocked",
        title="<Runtime title>",
        detail="Review <runtime detail>.",
        chat_enabled=False,
        can_load_models=False,
        checks=(),
        chat_model="<chat-model>",
        embedding_model="<embed-model>",
    )
    corpus = CorpusSnapshot(2, 7, 19, "ready")

    # Act
    rendered = render_sidebar_status(runtime=runtime, corpus=corpus)

    # Then
    assert all(
        value in rendered
        for value in (
            "Models",
            "Chat model",
            "&lt;chat-model&gt;",
            "Embedding model",
            "&lt;embed-model&gt;",
            "Corpus",
            "Ready",
            "2 documents",
            "7 pages",
            "19 chunks",
        )
    )
    assert "Runtime" not in rendered
    assert "Blocked" not in rendered
    assert "&lt;Runtime title&gt;" not in rendered
    assert "Review &lt;runtime detail&gt;." not in rendered
    assert "<Runtime title>" not in rendered
    assert "<chat-model>" not in rendered


def test_latest_evaluation_renders_compatible_facts_and_empty_state() -> None:
    # Arrange
    evaluation = EvaluationSummary(
        result_path="/private/result",
        split="<development>",
        systems=("dense", "<agentic>"),
        case_count=12,
        result_kind="standard",
        created_at="<2026-07-23>",
    )

    # Act
    rendered = render_latest_evaluation(evaluation)
    empty = render_latest_evaluation(None)

    # Then
    assert all(
        value in rendered
        for value in (
            "Standard benchmark",
            "&lt;development&gt;",
            "Dense, &lt;agentic&gt;",
            "12",
            "&lt;2026-07-23&gt;",
        )
    )
    assert "/private/result" not in rendered
    assert "No compatible standard benchmark is available." in empty


def test_shared_stylesheet_uses_semantic_visual_tokens_and_accessible_motion() -> None:
    # Arrange
    stylesheet = (UI_DIR / "app.css").read_text(encoding="utf-8")
    required_tokens = {
        "--canvas",
        "--surface",
        "--surface-raised",
        "--surface-hover",
        "--border",
        "--border-strong",
        "--text",
        "--text-muted",
        "--text-subtle",
        "--primary",
        "--primary-hover",
        "--focus",
        "--info",
        "--success",
        "--warning",
        "--danger",
    }

    # Assert the shared theme stays semantic and honours accessibility settings.
    assert all(f"{token}:" in stylesheet for token in required_tokens)
    assert "--overview-" not in stylesheet
    assert "--rag-" not in stylesheet
    assert "font-family: var(--font)" in stylesheet
    assert ":focus-visible" in stylesheet
    assert "min-height: 44px" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet


def test_shared_stylesheet_assigns_page_scroll_ownership() -> None:
    # Arrange
    stylesheet = (UI_DIR / "app.css").read_text(encoding="utf-8")

    # Assert each dense page keeps overflow within its designated region.
    assert "#ask-chatbot" in stylesheet
    assert "#ask-inspector" in stylesheet
    assert "overflow-y: auto" in stylesheet
    assert "#ask-composer" in stylesheet
    assert "position: sticky" in stylesheet
    assert "#inspector-document-inventory" in stylesheet
    assert ".evaluation-matrix" in stylesheet
    assert "overflow-x: auto" in stylesheet
    assert "#ask-header-actions" in stylesheet
    assert "#ask-evaluation-status" in stylesheet
    assert "#evaluation-results-layout" not in stylesheet
    assert "@media (max-width: 760px)" in stylesheet


def test_accessibility_script_enhances_scroll_regions_and_accordions() -> None:
    # Arrange
    script = (UI_DIR / "accessibility.js").read_text(encoding="utf-8")

    # Assert dynamic overflow remains keyboard-labelled after DOM updates.
    assert "ResizeObserver" in script
    assert 'target.setAttribute("role", "region")' in script
    assert 'target.setAttribute("tabindex", "0")' in script
    assert 'target.removeAttribute("tabindex")' in script
    assert "horizontally scrollable" in script
    assert "vertically scrollable" in script
    assert '"aria-expanded"' in script
    assert "MutationObserver" in script
