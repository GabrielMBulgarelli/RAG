"""Escaped HTML presenters for plain application data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any, Literal

from modules.ui.contracts import CorpusSnapshot, EvaluationSummary, RuntimeSnapshot
from modules.ui.evaluation_presenters import (
    EvaluationDashboardRender as EvaluationDashboardRender,
)
from modules.ui.evaluation_presenters import (
    render_evaluation_dashboard as render_evaluation_dashboard,
)
from modules.ui.system_presenter import render_system_page as render_system_page

StatusKind = Literal["info", "success", "warning", "error"]


def render_status(kind: StatusKind, title: str, detail: str = "") -> str:
    """Render an accessible status banner with escaped dynamic content."""
    icons = {"info": "i", "success": "✓", "warning": "!", "error": "×"}
    safe_title = escape(title)
    safe_detail = escape(detail)
    detail_html = f'<span class="rag-status__detail">{safe_detail}</span>' if detail else ""
    role = "alert" if kind == "error" else "status"
    live = "assertive" if kind == "error" else "polite"
    return (
        f'<div class="rag-status rag-status--{kind}" role="{role}" '
        f'aria-live="{live}" aria-atomic="true">'
        f'<span class="rag-status__icon" aria-hidden="true">{icons[kind]}</span>'
        '<span class="rag-status__copy">'
        f'<strong class="rag-status__title">{safe_title}</strong>{detail_html}'
        "</span></div>"
    )


def render_result_table(
    headers: Sequence[Any],
    rows: Sequence[Sequence[Any]] | None,
    *,
    caption: str,
    empty_message: str,
    mobile_cards: bool = False,
    table_class: str = "",
) -> str:
    """Render a safe read-only result table."""
    safe_headers = [escape(str(header)) for header in headers]
    safe_caption = escape(caption)
    safe_empty = escape(empty_message)
    row_values = list(rows or [])
    section_classes = "result-view"
    if table_class:
        section_classes += f" {escape(table_class)}"
    if not row_values:
        return (
            f'<section class="{section_classes}" aria-label="{safe_caption}">'
            f'<div class="result-empty" role="status">{safe_empty}</div>'
            "</section>"
        )

    status_kinds = {
        "ready": "success",
        "indexed": "success",
        "review": "warning",
        "warning": "warning",
        "unavailable": "error",
        "error": "error",
        "failed": "error",
        "not loaded": "info",
        "pending": "info",
    }
    card_class = " stack-on-mobile" if mobile_cards else ""
    body = []
    for row in row_values:
        cells = []
        for index, header in enumerate(safe_headers):
            raw_value = row[index] if index < len(row) else "—"
            display_value = "—" if raw_value is None or raw_value == "" else str(raw_value)
            safe_value = escape(display_value)
            status = status_kinds.get(display_value.strip().lower())
            status_attribute = f' data-status="{status}"' if status else ""
            cells.append(f'<td data-label="{header}"{status_attribute}>{safe_value}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")

    headings = "".join(f'<th scope="col">{header}</th>' for header in safe_headers)
    return (
        f'<section class="{section_classes}">'
        '<div class="result-scroll">'
        f'<table class="result-table{card_class}">'
        f"<caption>{safe_caption}</caption>"
        f"<thead><tr>{headings}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div></section>"
    )


def render_evidence(sources: Sequence[Mapping[str, Any]]) -> str:
    """Render public citation fields without leaking unrelated result data."""
    if not sources:
        return (
            '<section class="evidence-list evidence-list--empty" '
            'aria-label="Cited evidence">'
            '<p class="empty-result">No cited evidence yet. Sources will appear after a '
            "supported answer.</p></section>"
        )
    items = []
    for source in sources:
        label = escape(str(source.get("label") or "Source"))
        filename = escape(str(source.get("filename") or "Unknown file"))
        page = source.get("page")
        location = f"{filename} · Page {escape(str(page))}" if page is not None else filename
        excerpt = escape(str(source.get("excerpt") or "No excerpt available."))
        relevant_badge = (
            '<span class="evidence-state evidence-state--relevant">Relevant</span>'
            if source.get("relevant", True)
            else ""
        )
        items.append(
            '<article class="evidence-item"><div class="evidence-item__header">'
            f'<span class="evidence-citation">{label}</span>'
            f'<strong class="evidence-location">{location}</strong>'
            '<span class="evidence-state evidence-state--cited">Cited</span>'
            f"{relevant_badge}</div>"
            f'<blockquote class="evidence-excerpt">{excerpt}</blockquote></article>'
        )
    return (
        '<section class="evidence-list" aria-label="Cited evidence">'
        + "".join(items)
        + "</section>"
    )


def render_corpus_summary(corpus: CorpusSnapshot) -> str:
    return (
        '<section class="document-summary" role="status" aria-live="polite">'
        f'<span data-state="{escape(corpus.status)}">{escape(_state_label(corpus.status))}</span>'
        '<div class="document-summary__counts">'
        f"<strong>{_count(value=corpus.document_count, singular='document')}</strong>"
        f"<strong>{_count(value=corpus.page_count, singular='page')}</strong>"
        f"<strong>{_count(value=corpus.chunk_count, singular='chunk')}</strong>"
        "</div></section>"
    )


def render_selected_document(*, relative_path: str, page_count: int, chunk_count: int) -> str:
    return (
        '<section class="document-selected" aria-label="Selected document">'
        f"<strong>{escape(relative_path)}</strong>"
        f"<span>{_count(value=page_count, singular='page')} · "
        f"{_count(value=chunk_count, singular='chunk')}</span>"
        "</section>"
    )


def render_indexing_errors(rows: Sequence[Sequence[Any]] | None) -> str:
    return render_result_table(
        ["Document", "Operation", "Error type", "Message"],
        rows,
        caption="Indexing errors",
        empty_message="No indexing errors.",
        mobile_cards=True,
        table_class="document-indexing-errors",
    )


def _count(*, value: int, singular: str, plural: str | None = None) -> str:
    return f"{value} {singular if value == 1 else plural or singular + 's'}"


def _state_label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def render_latest_evaluation(evaluation: EvaluationSummary | None) -> str:
    """Render the latest compatible standard benchmark without exposing its path."""
    return (
        '<p class="latest-evaluation-empty">No compatible standard benchmark is available.</p>'
        if evaluation is None
        else (
            '<dl class="latest-evaluation-facts">'
            f"<div><dt>Type</dt><dd>{'Standard benchmark' if evaluation.result_kind == 'standard' else 'Custom evaluation'}</dd></div>"
            f"<div><dt>Split</dt><dd>{escape(evaluation.split)}</dd></div>"
            f"<div><dt>Systems</dt><dd>{escape(', '.join(_state_label(system) for system in evaluation.systems) or '—')}</dd></div>"
            f"<div><dt>Cases</dt><dd>{evaluation.case_count}</dd></div>"
            f"<div><dt>Result date</dt><dd>{escape(evaluation.created_at or '—')}</dd></div>"
            "</dl>"
        )
    )


def render_sidebar_status(*, runtime: RuntimeSnapshot, corpus: CorpusSnapshot) -> str:
    """Render escaped model names and complete Corpus state for the shared sidebar."""
    return (
        '<section class="sidebar-status" aria-label="Application status" '
        'aria-live="polite" aria-atomic="true">'
        '<article class="sidebar-card">'
        '<div class="sidebar-card__heading"><h2>Models</h2></div>'
        '<dl class="sidebar-facts">'
        f"<div><dt>Chat model</dt><dd>{escape(runtime.chat_model or '—')}</dd></div>"
        f"<div><dt>Embedding model</dt><dd>{escape(runtime.embedding_model or '—')}</dd></div>"
        "</dl></article>"
        '<article class="sidebar-card">'
        '<div class="sidebar-card__heading"><h2>Corpus</h2>'
        f'<span data-state="{escape(corpus.status)}">{escape(_state_label(corpus.status))}</span>'
        "</div>"
        '<div class="sidebar-counts">'
        f"<strong>{_count(value=corpus.document_count, singular='document')}</strong>"
        f"<strong>{_count(value=corpus.page_count, singular='page')}</strong>"
        f"<strong>{_count(value=corpus.chunk_count, singular='chunk')}</strong>"
        "</div></article></section>"
    )
