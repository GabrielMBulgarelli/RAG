"""Shared multipage shell for the Gradio application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
from gradio.components.navbar import Navbar

from modules.ui.pages.ask import build_ask_page
from modules.ui.pages.document_components import (
    SidebarDocumentComponents,
    build_sidebar_document_components,
)
from modules.ui.pages.documents import bind_document_management
from modules.ui.presenters import render_sidebar_status

if TYPE_CHECKING:
    from modules.ui.application import RAGApplication

_APP_STYLESHEET = Path(__file__).with_name("app.css")
_ACCESSIBILITY_SCRIPT = Path(__file__).with_name("accessibility.js").read_text(encoding="utf-8")


def render_navigation() -> NavigationComponents:
    """Render the document sidebar and return its live status surface."""
    with gr.Sidebar(
        label="Application navigation",
        open=True,
        width=280,
        elem_id="application-sidebar",
    ):
        gr.HTML(
            """
            <header>
              <strong>Local Document RAG</strong>
              <span>AI engineering dashboard</span>
            </header>
            """
        )
        documents = build_sidebar_document_components()
        status = gr.HTML(
            '<section class="sidebar-status" aria-label="Application status" role="status">'
            "Loading Models and Corpus…</section>",
            elem_id="sidebar-status",
        )
    return NavigationComponents(documents, status)


@dataclass(frozen=True)
class NavigationComponents:
    documents: SidebarDocumentComponents
    status: gr.HTML


def _render_page(*, page_name: str, application: RAGApplication) -> None:
    Navbar(visible=False, main_page_name=False)
    gr.HTML(
        '<a class="skip-link" href="#page-content">Skip to page content</a>',
        elem_id="skip-navigation",
    )
    navigation = render_navigation()
    refresh_sidebar: Callable[[], str] = partial(
        lambda app: render_sidebar_status(
            runtime=app.runtime_snapshot(),
            corpus=app.corpus_snapshot(),
        ),
        application,
    )
    gr.on(
        triggers=None,
        fn=refresh_sidebar,
        outputs=navigation.status,
        show_progress="hidden",
    )
    if page_name == "Ask Documents":
        ask = build_ask_page(
            application,
            sidebar_status=navigation.status,
            refresh_sidebar=refresh_sidebar,
        )
        bind_document_management(
            application=application,
            sidebar=navigation.documents,
            sidebar_status=navigation.status,
            refresh_sidebar=refresh_sidebar,
            indexed_documents=ask.inspector.documents,
            corpus_context=ask.conversation.corpus,
        )
    else:
        gr.Markdown(f"# {page_name}")


def build_application(application: RAGApplication) -> gr.Blocks:
    """Build the Ask Documents application at the root route."""
    with gr.Blocks(
        title="Local Document RAG",
        css_paths=_APP_STYLESHEET,
        js=_ACCESSIBILITY_SCRIPT,
        fill_width=True,
        fill_height=True,
    ) as interface:
        _render_page(page_name="Ask Documents", application=application)

    return interface
