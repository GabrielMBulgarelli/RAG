"""Presentation helpers for the analytical evaluation dashboard."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from modules.ui.contracts import (
    EvaluationPageSnapshot,
    EvaluationSummary,
)


@dataclass(frozen=True)
class EvaluationDashboardRender:
    """HTML fragments and control state for the analytical evaluation page."""

    context: str
    matrix: str
    empty_state: str


def _state_label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _render_context(summary: EvaluationSummary | None) -> str:
    if summary is None:
        return ""
    result_label = (
        "Standard benchmark" if summary.result_kind == "standard" else "Custom evaluation"
    )
    facts = (
        ("Type", result_label),
        ("Split", summary.split),
        ("Systems", ", ".join(_state_label(system) for system in summary.systems) or "—"),
        ("Model", summary.chat_model),
        ("Cases", str(summary.case_count)),
        ("Result date", summary.created_at or "—"),
    )
    return (
        '<section class="evaluation-context" aria-label="Evaluation result context">'
        + "".join(
            '<div class="evaluation-context__item">'
            f"<span>{escape(label)}</span><strong>{escape(value)}</strong></div>"
            for label, value in facts
        )
        + "</section>"
    )


def _render_matrix(rows: Sequence[Sequence[Any]]) -> str:
    headers = ("Category", "Metric", "Dense", "BM25", "Hybrid", "Agentic")
    body = []
    for row in rows:
        cells = [
            f'<th scope="row" data-label="{headers[0]}">{escape(str(row[0]))}</th>',
            f'<td data-label="{headers[1]}">{escape(str(row[1]))}</td>',
        ]
        for index, header in enumerate(headers[2:], start=2):
            raw_value = str(row[index]) if index < len(row) else "—"
            if raw_value == "—":
                primary, support = "—", "Missing observation"
            elif raw_value.startswith("— "):
                primary, support = "—", raw_value[2:]
            elif " · " in raw_value:
                primary, support = raw_value.split(" · ", 1)
            else:
                primary, support = raw_value, ""
            support_html = (
                f'<span class="metric-support">{escape(support)}</span>' if support else ""
            )
            cells.append(
                f'<td data-label="{header}"><span class="metric-value">{escape(primary)}</span>'
                f"{support_html}</td>"
            )
        body.append(f"<tr>{''.join(cells)}</tr>")
    headings = "".join(f'<th scope="col">{header}</th>' for header in headers)
    empty = (
        '<tr><td colspan="6" class="evaluation-muted">No metric observations available.</td></tr>'
        if not body
        else ""
    )
    return (
        '<section class="result-view evaluation-metrics-view">'
        '<div class="result-scroll" tabindex="0" aria-label="Evaluation metrics comparison">'
        '<table class="result-table evaluation-matrix"><caption>Exact metric matrix</caption>'
        f"<thead><tr>{headings}</tr></thead><tbody>{''.join(body)}{empty}</tbody>"
        "</table></div></section>"
    )


def _render_empty(snapshot: EvaluationPageSnapshot) -> str:
    if snapshot.latest is not None:
        return ""
    detail = (
        snapshot.problems[0]
        if snapshot.problems
        else "Run the standard benchmark to create a complete schema version 2 result."
    )
    kind = "warning" if snapshot.state == "blocked" else "info"
    return (
        f'<section class="status-banner status-banner--{kind}" role="status" aria-live="polite">'
        '<span class="status-icon" aria-hidden="true">!</span><div>'
        f"<strong>No compatible standard benchmark</strong><span>{escape(detail)}</span>"
        "</div></section>"
    )


def render_evaluation_dashboard(
    snapshot: EvaluationPageSnapshot,
) -> EvaluationDashboardRender:
    """Render a result-first dashboard without aggregating unlike metrics."""
    latest = snapshot.latest
    return EvaluationDashboardRender(
        context=_render_context(latest),
        matrix=_render_matrix(snapshot.metric_rows),
        empty_state=_render_empty(snapshot),
    )
