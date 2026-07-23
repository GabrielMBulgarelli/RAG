"""Evidence-first document question workspace."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import gradio as gr

from modules.ui.pages.ask_callbacks import (
    begin_evaluation,
    clear_conversation,
    export_conversation,
    load_models_and_refresh_ask,
    refresh_ask_readiness,
    run_default_evaluation,
    submit_question,
)
from modules.ui.pages.ask_components import AskComponents, build_ask_components

if TYPE_CHECKING:
    from modules.ui.application import RAGApplication


def _bind_readiness(
    *,
    application: RAGApplication,
    components: AskComponents,
    sidebar_status: gr.HTML | None = None,
    refresh_sidebar: Callable[[], str] | None = None,
) -> None:
    conversation = components.conversation
    composer = conversation.composer
    outputs = [
        conversation.corpus,
        components.model_status,
        composer.question,
        composer.ask,
        composer.load_models,
    ]
    gr.on(
        triggers=None,
        fn=lambda: refresh_ask_readiness(application),
        outputs=outputs,
        show_progress="hidden",
    )
    load_event = composer.load_models.click(
        fn=lambda: load_models_and_refresh_ask(application),
        outputs=outputs,
        show_progress="minimal",
    )
    if sidebar_status is not None and refresh_sidebar is not None:
        load_event.then(fn=refresh_sidebar, outputs=sidebar_status, show_progress="hidden")


def _bind_questions(
    *,
    application: RAGApplication,
    components: AskComponents,
) -> None:
    conversation = components.conversation
    composer = conversation.composer
    inspector = components.inspector
    inputs = [composer.question, conversation.chatbot, components.session_id]
    outputs = [
        composer.question,
        conversation.chatbot,
        components.latest_result,
        conversation.answer_state,
        inspector.sources,
        inspector.retrieval,
        inspector.trace,
        inspector.raw_trace,
        inspector.query,
    ]
    composer.ask.click(
        fn=lambda message, history, session_id: submit_question(
            application=application,
            message=message,
            history=history,
            session_id=session_id,
        ),
        inputs=inputs,
        outputs=outputs,
        concurrency_limit=1,
        api_name="ask_with_button",
    )
    composer.question.submit(
        fn=lambda message, history, session_id: submit_question(
            application=application,
            message=message,
            history=history,
            session_id=session_id,
        ),
        inputs=inputs,
        outputs=outputs,
        concurrency_limit=1,
        api_name="ask_with_enter",
    )


def _bind_actions(
    *,
    application: RAGApplication,
    components: AskComponents,
) -> None:
    conversation = components.conversation
    composer = conversation.composer
    inspector = components.inspector
    composer.clear.click(
        fn=lambda session_id: clear_conversation(
            application=application,
            session_id=session_id,
        ),
        inputs=components.session_id,
        outputs=[
            conversation.chatbot,
            components.latest_result,
            conversation.answer_state,
            inspector.sources,
            inspector.retrieval,
            inspector.trace,
            inspector.raw_trace,
            inspector.query,
            components.session_id,
        ],
    )
    composer.export.click(
        fn=lambda messages, result: export_conversation(
            application=application,
            messages=messages,
            result=result,
        ),
        inputs=[conversation.chatbot, components.latest_result],
        outputs=composer.download,
    )


def _bind_evaluation(
    *,
    application: RAGApplication,
    components: AskComponents,
) -> None:
    evaluation = components.evaluation
    evaluation.run.click(
        fn=begin_evaluation,
        outputs=[evaluation.run, evaluation.status],
        queue=False,
    ).then(
        fn=partial(run_default_evaluation, application),
        outputs=[evaluation.run, evaluation.status],
        show_progress="full",
        concurrency_limit=1,
    )


def build_ask_page(
    application: RAGApplication,
    *,
    sidebar_status: gr.HTML | None = None,
    refresh_sidebar: Callable[[], str] | None = None,
) -> AskComponents:
    """Create the primary AI workspace and bind existing controller behavior."""
    components = build_ask_components()
    _bind_readiness(
        application=application,
        components=components,
        sidebar_status=sidebar_status,
        refresh_sidebar=refresh_sidebar,
    )
    _bind_questions(application=application, components=components)
    _bind_actions(application=application, components=components)
    _bind_evaluation(application=application, components=components)
    return components
