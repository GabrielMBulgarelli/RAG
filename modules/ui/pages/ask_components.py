"""Gradio component construction for the Ask workspace."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import gradio as gr

from modules.ui.ask_presenters import (
    InspectorContent,
    render_query_details,
    render_query_retrieval,
    render_query_sources,
    render_query_trace,
)
from modules.ui.contracts import QueryDiagnostics
from modules.ui.pages.document_components import (
    IndexedDocumentComponents,
    build_indexed_document_components,
)


@dataclass(frozen=True)
class ComposerComponents:
    question: gr.Textbox
    ask: gr.Button
    load_models: gr.Button
    clear: gr.Button
    export: gr.Button
    download: gr.DownloadButton


@dataclass(frozen=True)
class ConversationComponents:
    corpus: gr.HTML
    chatbot: gr.Chatbot
    answer_state: gr.HTML
    composer: ComposerComponents


@dataclass(frozen=True)
class InspectorComponents:
    documents: IndexedDocumentComponents
    sources: gr.HTML
    retrieval: gr.HTML
    trace: gr.HTML
    raw_trace: gr.HTML
    query: gr.HTML


@dataclass(frozen=True)
class EvaluationActionComponents:
    run: gr.Button
    status: gr.HTML


@dataclass(frozen=True)
class AskComponents:
    session_id: gr.State
    latest_result: gr.State
    model_status: gr.HTML
    evaluation: EvaluationActionComponents
    conversation: ConversationComponents
    inspector: InspectorComponents


def render_model_status(*, loaded: bool) -> str:
    """Render the compact Ask-page model readiness indicator."""
    modifier = "loaded" if loaded else "not-loaded"
    label = "Model Loaded" if loaded else "Model Not Loaded"
    return (
        f'<div class="ask-model-status ask-model-status--{modifier}" '
        'role="status" aria-live="polite" aria-atomic="true">'
        '<span class="ask-model-status__dot" aria-hidden="true"></span>'
        f"<span>{label}</span>"
        "</div>"
    )


def empty_inspector_content() -> InspectorContent:
    """Create the initial, presentation-neutral inspector content."""
    timeline, raw_trace = render_query_trace(())
    query = render_query_details(
        QueryDiagnostics("", "", (), 0, "", "", "not_reported"),
        original_question="",
        standalone_question="",
        retrieval_rounds=0,
    )
    return InspectorContent(
        sources=render_query_sources(()),
        retrieval=render_query_retrieval(()),
        timeline=timeline,
        raw_trace=raw_trace,
        query=query,
    )


def _build_composer() -> ComposerComponents:
    with gr.Group(elem_id="ask-composer"):
        question = gr.Textbox(
            label="Question",
            placeholder="Load AI models before asking about your documents",
            interactive=False,
            lines=3,
            max_lines=6,
            elem_id="ask-question",
        )
        with gr.Row(elem_classes="ask-action-row"):
            ask = gr.Button("Ask", variant="primary", interactive=False)
            load_models = gr.Button("Load AI Models", variant="secondary")
            clear = gr.Button("Clear", variant="secondary")
            export = gr.Button("Export", variant="secondary")
            download = gr.DownloadButton(
                "Download export",
                visible=False,
                interactive=False,
                elem_id="ask-download",
            )
    return ComposerComponents(question, ask, load_models, clear, export, download)


def _build_conversation() -> ConversationComponents:
    corpus = gr.HTML(
        '<div class="ask-corpus-context" role="status">Loading corpus…</div>',
        elem_id="ask-corpus-context",
    )
    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages",
        allow_tags=False,
        height=500,
        elem_id="ask-chatbot",
    )
    answer_state = gr.HTML("", visible=False, elem_id="ask-answer-state")
    return ConversationComponents(
        corpus,
        chatbot,
        answer_state,
        _build_composer(),
    )


def _build_inspector(content: InspectorContent) -> InspectorComponents:
    gr.Markdown("## Inspector")
    with gr.Tabs():
        with gr.Tab("Indexed Documents"):
            documents = build_indexed_document_components()
        with gr.Tab("Sources"):
            sources = gr.HTML(
                content.sources,
                elem_id="ask-sources",
                elem_classes=["overflow-region"],
            )
        with gr.Tab("Retrieval"):
            retrieval = gr.HTML(
                content.retrieval,
                elem_id="ask-retrieval",
                elem_classes=["overflow-region"],
            )
        with gr.Tab("Trace"):
            trace = gr.HTML(content.timeline, elem_id="ask-trace")
            with gr.Accordion("Raw details", open=False):
                raw_trace = gr.HTML(
                    content.raw_trace,
                    elem_id="ask-raw-trace",
                    elem_classes=["overflow-region"],
                )
        with gr.Tab("Query"):
            query = gr.HTML(content.query, elem_id="ask-query")
    return InspectorComponents(documents, sources, retrieval, trace, raw_trace, query)


def build_ask_components() -> AskComponents:
    """Create the Ask workspace component tree without binding behavior."""
    session_id = gr.State(lambda: str(uuid4()))
    latest_result = gr.State({})
    content = empty_inspector_content()
    with gr.Column(
        elem_id="page-content",
        elem_classes=["app-page", "app-page--ask"],
    ):
        with gr.Row(elem_classes=["ask-page-header"]):
            gr.Markdown("# Ask Documents", elem_classes=["ask-page-title"])
            with gr.Row(elem_id="ask-header-actions"):
                model_status = gr.HTML(
                    render_model_status(loaded=False),
                    elem_id="ask-model-status",
                    padding=False,
                )
                run_evaluation = gr.Button(
                    "Run evaluation",
                    variant="primary",
                    elem_id="run-evaluation",
                )
        evaluation_status = gr.HTML(
            "",
            visible=False,
            elem_id="ask-evaluation-status",
        )
        gr.Markdown("Ask grounded questions and inspect the evidence, retrieval, and public trace.")
        with gr.Row(equal_height=False, elem_id="ask-workspace"):
            with gr.Column(scale=3, min_width=0, elem_id="ask-conversation"):
                conversation = _build_conversation()
            with gr.Column(scale=2, min_width=340, elem_id="ask-inspector"):
                inspector = _build_inspector(content)
    return AskComponents(
        session_id,
        latest_result,
        model_status,
        EvaluationActionComponents(
            run_evaluation,
            evaluation_status,
        ),
        conversation,
        inspector,
    )
