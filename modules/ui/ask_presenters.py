"""Escaped presenters for the evidence-first Ask workspace."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from modules.ui.contracts import (
    AnswerState,
    QueryDiagnostics,
    RetrievalHitView,
    SourceView,
    TraceEventView,
)
from modules.ui.presenters import (
    StatusKind,
    render_evidence,
    render_result_table,
    render_status,
)

TraceRow = tuple[
    str,
    str,
    int | None,
    int | None,
    int | None,
    int,
    int,
    str,
    str,
]


@dataclass(frozen=True)
class InspectorContent:
    sources: str
    retrieval: str
    timeline: str
    raw_trace: str
    query: str


def render_answer_state(state: AnswerState) -> str:
    """Render the latest public answer support state."""
    states: dict[AnswerState, tuple[StatusKind, str, str]] = {
        "supported": (
            "success",
            "Supported",
            "The answer is backed by sufficient cited evidence.",
        ),
        "limited": (
            "warning",
            "Limited",
            "Only part of the answer is supported by the available evidence.",
        ),
        "abstention": (
            "warning",
            "Abstention",
            "The indexed evidence is insufficient for a grounded answer.",
        ),
        "unavailable": (
            "error",
            "Unavailable",
            "The local RAG service could not complete this question.",
        ),
        "completed": (
            "info",
            "Completed",
            "Review the evidence and query details for this answer.",
        ),
    }
    kind, title, detail = states[state]
    return render_status(kind, title, detail)


def render_query_sources(sources: Sequence[SourceView]) -> str:
    """Render cited source records using only their public fields."""
    return render_evidence(
        [
            {
                "label": source.label,
                "filename": source.filename,
                "page": source.page,
                "excerpt": source.excerpt,
                "relevant": True,
            }
            for source in sources
        ]
    )


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_query_retrieval(hits: Sequence[RetrievalHitView]) -> str:
    """Render the complete public retrieval score decomposition."""
    rows = [
        (
            hit.chunk_id,
            hit.filename,
            hit.page,
            _score(hit.semantic_score),
            _score(hit.sparse_score),
            _score(hit.fused_score),
            _score(hit.selection_score),
            ", ".join(hit.matched_subqueries) or "—",
        )
        for hit in hits
    ]
    return render_result_table(
        (
            "Chunk ID",
            "Filename",
            "Page",
            "Semantic",
            "Sparse",
            "Fused",
            "Selection",
            "Matched subqueries",
        ),
        rows,
        caption="Retrieval scores",
        empty_message="Retrieval details will appear after a question.",
        mobile_cards=True,
        table_class="ask-retrieval-table",
    )


def _display(value: str) -> str:
    return value.replace("_", " ").strip().title() or "—"


def _trace_rows(trace: Sequence[TraceEventView]) -> list[TraceRow]:
    return [
        (
            _display(event.stage),
            _display(event.decision),
            event.retrieved_count,
            event.fused_count,
            event.selected_count,
            event.retry_count,
            event.llm_calls,
            _display(event.termination),
            "—" if event.duration_ms is None else f"{event.duration_ms:.0f} ms",
        )
        for event in trace
    ]


def _render_trace_event(event: TraceEventView) -> str:
    facts = (
        ("Retrieved", event.retrieved_count),
        ("Fused", event.fused_count),
        ("Selected", event.selected_count),
        ("Retries", event.retry_count),
        ("Model calls", event.llm_calls),
        ("Termination", _display(event.termination)),
        (
            "Duration",
            "—" if event.duration_ms is None else f"{event.duration_ms:.0f} ms",
        ),
    )
    fact_html = "".join(
        f"<li><span>{escape(label)}</span><strong>"
        f"{escape(str(value if value is not None else '—'))}</strong></li>"
        for label, value in facts
    )
    return (
        '<article class="trace-event">'
        '<div class="trace-event__marker" aria-hidden="true"></div>'
        f'<div><span class="trace-event__stage">{escape(_display(event.stage))}</span>'
        f"<h4>{escape(_display(event.decision))}</h4>"
        f"<ul>{fact_html}</ul></div></article>"
    )


def _render_trace_timeline(trace: Sequence[TraceEventView]) -> str:
    if not trace:
        return (
            '<div class="result-empty" role="status">'
            "The public trace will appear after a question.</div>"
        )
    return (
        '<section class="trace-timeline" aria-label="Public query trace">'
        + "".join(_render_trace_event(event) for event in trace)
        + "</section>"
    )


def render_query_trace(trace: Sequence[TraceEventView]) -> tuple[str, str]:
    """Render a scan-friendly public timeline plus its exact tabular details."""
    raw = render_result_table(
        (
            "Stage",
            "Decision",
            "Retrieved",
            "Fused",
            "Selected",
            "Retry",
            "LLM calls",
            "Termination",
            "Duration",
        ),
        _trace_rows(trace),
        caption="Raw public trace",
        empty_message="No trace details yet.",
        mobile_cards=True,
        table_class="ask-trace-table",
    )
    return _render_trace_timeline(trace), raw


def render_query_details(
    diagnostics: QueryDiagnostics,
    *,
    original_question: str,
    standalone_question: str,
    retrieval_rounds: int,
) -> str:
    """Render the public routing and evidence diagnostic fields."""
    facts = (
        ("Original question", original_question or "—"),
        ("Standalone question", standalone_question or "—"),
        ("Route", _display(diagnostics.route)),
        ("Strategy", _display(diagnostics.retrieval_strategy)),
        ("Subqueries", ", ".join(diagnostics.subqueries) or "None"),
        ("Retrieval rounds", retrieval_rounds),
        ("Retries", diagnostics.retry_count),
        ("Evidence", _display(diagnostics.evidence_state)),
        ("Conflict", _display(diagnostics.conflict_state)),
        ("Citation validation", _display(diagnostics.citation_validation)),
    )
    return (
        '<dl class="query-details">'
        + "".join(
            f"<div><dt>{escape(label)}</dt><dd>{escape(str(value))}</dd></div>"
            for label, value in facts
        )
        + "</dl>"
    )
