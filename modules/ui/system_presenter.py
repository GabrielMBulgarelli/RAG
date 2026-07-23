"""Escaped HTML presenter for the System diagnostics page."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from modules.ui.contracts import SystemCheck, SystemPageSnapshot


def _state_label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _checks(*, title: str, items: Sequence[SystemCheck]) -> str:
    rows = "".join(
        '<li class="system-check" data-state="{state}">'
        '<span class="system-check__state">{label}</span>'
        '<span class="system-check__copy"><strong>{name}</strong>'
        "<small>{detail}</small></span></li>".format(
            state=escape(item.state),
            label=escape(_state_label(item.state)),
            name=escape(item.name),
            detail=escape(item.detail),
        )
        for item in items
    )
    return (
        '<section class="system-panel">'
        f"<h2>{escape(title)}</h2>"
        f'<ul class="system-checks">{rows}</ul></section>'
    )


def _summary(snapshot: SystemPageSnapshot) -> str:
    if snapshot.state == "blocked":
        return (
            '<div class="rag-status rag-status--error" role="alert" '
            'aria-live="assertive" aria-atomic="true">'
            '<span class="rag-status__icon" aria-hidden="true">×</span>'
            '<span class="rag-status__copy">'
            f'<strong class="rag-status__title">{escape(snapshot.title)}</strong>'
            f'<span class="rag-status__detail">{escape(snapshot.detail)}</span>'
            "</span></div>"
        )
    return (
        f'<section class="system-summary" data-state="{escape(snapshot.state)}" '
        'role="status" aria-live="polite">'
        f"<strong>{escape(snapshot.title)}</strong><span>{escape(snapshot.detail)}</span>"
        "</section>"
    )


def render_system_page(snapshot: SystemPageSnapshot) -> str:
    """Render grouped, escaped diagnostics without exposing private configuration."""
    configuration = "".join(
        "<div><dt>{name}</dt><dd>{value}</dd></div>".format(
            name=escape(item.name),
            value=escape(item.value),
        )
        for item in snapshot.safe_configuration
    )
    return (
        '<main class="system-dashboard" aria-label="System diagnostics">'
        f"{_summary(snapshot)}"
        f"{_checks(title='AI runtime', items=snapshot.runtime_checks)}"
        f"{_checks(title='Document index', items=snapshot.index_checks)}"
        f"{_checks(title='Evaluation', items=snapshot.evaluation_checks)}"
        '<section class="system-panel system-configuration">'
        "<h2>Safe configuration</h2>"
        f"<dl>{configuration}</dl></section></main>"
    )
