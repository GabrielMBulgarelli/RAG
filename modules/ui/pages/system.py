"""System diagnostics page."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import gradio as gr

from modules.ui.contracts import SystemPageSnapshot
from modules.ui.events import ComponentUpdate, system_control_update
from modules.ui.presenters import render_system_page

if TYPE_CHECKING:
    from modules.ui.application import RAGApplication


def system_values(snapshot: SystemPageSnapshot) -> tuple[str, ComponentUpdate]:
    """Translate a fresh System snapshot into page component values."""
    return render_system_page(snapshot), system_control_update(snapshot)


def refresh_system(application: RAGApplication) -> tuple[str, ComponentUpdate]:
    return system_values(application.system_snapshot())


def load_models_and_refresh(application: RAGApplication) -> tuple[str, ComponentUpdate]:
    application.load_ai_models()
    return refresh_system(application)


def build_system_page(  # lanorme: ignore[SIZE-002] -- Declarative page composition
    application: RAGApplication,
    *,
    sidebar_status: gr.HTML | None = None,
    refresh_sidebar: Callable[[], str] | None = None,
) -> None:
    """Create the runtime, index, evaluation, and safe-configuration dashboard."""
    with gr.Column(
        elem_id="page-content",
        elem_classes=["app-page", "app-page--system"],
    ):
        gr.Markdown("# System")
        gr.Markdown(
            "Inspect local runtime readiness, index integrity, and evaluation dependencies."
        )
        content = gr.HTML(
            '<div role="status">Loading system diagnostics…</div>',
            elem_id="system-diagnostics",
        )
        with gr.Row(elem_id="system-actions"):
            load_models = gr.Button(
                "Load AI Models",
                variant="primary",
                visible=False,
                elem_id="system-load-models",
            )
            refresh = gr.Button(
                "Refresh",
                variant="secondary",
                elem_id="system-refresh",
            )
        gr.HTML("", visible=False, elem_id="system-status")

        gr.on(
            triggers=None,
            fn=partial(refresh_system, application),
            outputs=[content, load_models],
            show_progress="hidden",
        )
        refresh_event = refresh.click(
            fn=partial(refresh_system, application),
            outputs=[content, load_models],
            show_progress="minimal",
        )
        load_event = load_models.click(
            fn=partial(load_models_and_refresh, application),
            outputs=[content, load_models],
            show_progress="minimal",
        )
        if sidebar_status is not None and refresh_sidebar is not None:
            for event in (refresh_event, load_event):
                event.then(fn=refresh_sidebar, outputs=sidebar_status, show_progress="hidden")
