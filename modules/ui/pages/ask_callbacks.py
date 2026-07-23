"""Typed controller adapters for the Ask workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from uuid import uuid4

from modules.config import config
from modules.ui.ask_presenters import (
    InspectorContent,
    render_answer_state,
    render_query_details,
    render_query_retrieval,
    render_query_sources,
    render_query_trace,
)
from modules.ui.contracts import QuerySnapshot
from modules.ui.events import ComponentUpdate, ask_control_updates, component_update
from modules.ui.pages.ask_components import render_model_status
from modules.ui.presenters import render_status

if TYPE_CHECKING:
    from modules.ui.application import RAGApplication

ChatMessage = dict[str, str]
ChatHistory = list[ChatMessage]
PublicResult = dict[str, object]
ReadinessValues = tuple[
    str,
    ComponentUpdate,
    ComponentUpdate,
    ComponentUpdate,
    ComponentUpdate,
]
SubmitValues = tuple[
    str,
    ChatHistory,
    PublicResult,
    ComponentUpdate,
    str,
    str,
    str,
    str,
    str,
]
ClearValues = tuple[
    ChatHistory,
    PublicResult,
    ComponentUpdate,
    str,
    str,
    str,
    str,
    str,
    str,
]
_EVALUATION_SYSTEMS = ("dense", "bm25", "hybrid", "agentic")


def run_default_evaluation(
    application: RAGApplication,
) -> tuple[ComponentUpdate, ComponentUpdate]:
    """Run the standard evaluator and return the Ask-page action state."""
    snapshot = application.run_evaluation_snapshot(
        split="development",
        systems=list(_EVALUATION_SYSTEMS),
        chat_model=config.llm_model,
    )
    if snapshot.state == "error":
        status = render_status(
            "error",
            "Evaluation could not run",
            snapshot.problems[0] if snapshot.problems else "Review the evaluation configuration.",
        )
    elif snapshot.state == "blocked":
        status = render_status(
            "warning",
            "Evaluation unavailable",
            snapshot.problems[0] if snapshot.problems else "Review system readiness.",
        )
    elif snapshot.state == "saved_result" and snapshot.latest is not None:
        status = render_status(
            "success",
            "Evaluation complete",
            "The result was saved successfully.",
        )
    else:
        status = render_status(
            "warning",
            "Evaluation unavailable",
            snapshot.problems[0]
            if snapshot.problems
            else "No evaluation result was saved. Review system readiness.",
        )
    return (
        cast(ComponentUpdate, component_update(interactive=True)),
        cast(ComponentUpdate, component_update(value=status, visible=True)),
    )


def begin_evaluation() -> tuple[ComponentUpdate, ComponentUpdate]:
    """Disable the evaluation action while the standard run is active."""
    return (
        cast(ComponentUpdate, component_update(interactive=False)),
        cast(
            ComponentUpdate,
            component_update(
                value=render_status(
                    "info",
                    "Running evaluationâ€¦",
                    "This may take several minutes for systems that use the local AI model.",
                ),
                visible=True,
            ),
        ),
    )


def render_corpus_context(application: RAGApplication) -> str:
    corpus = application.corpus_snapshot()
    return (
        '<div class="ask-corpus-context" role="status" aria-live="polite">'
        f"<strong>{corpus.document_count} document(s)</strong>"
        f"<span>{corpus.page_count} page(s) · {corpus.chunk_count} chunk(s)</span>"
        "</div>"
    )


def refresh_ask_readiness(application: RAGApplication) -> ReadinessValues:
    """Load current backend readiness independently for this routed page."""
    runtime = application.runtime_snapshot()
    composer, submit, load_models = ask_control_updates(runtime)
    model_status = cast(
        ComponentUpdate,
        component_update(
            value=render_model_status(loaded=runtime.chat_enabled),
            visible=True,
        ),
    )
    return render_corpus_context(application), model_status, composer, submit, load_models


def load_models_and_refresh_ask(application: RAGApplication) -> ReadinessValues:
    application.load_ai_models()
    return refresh_ask_readiness(application)


def _public_result(
    *,
    application: RAGApplication,
    messages: ChatHistory,
    result: PublicResult,
) -> PublicResult:
    public = application.public_export(messages, result)
    return {
        "standalone_query": public.get("standalone_query"),
        "route": public.get("route"),
        "strategy": public.get("strategy"),
        "subqueries": public.get("subqueries", []),
        "retry_count": public.get("retry_count", 0),
        "evidence_status": public.get("evidence_status"),
        "sources": public.get("citations", []),
        "validation": public.get("validation"),
        "trace": public.get("public_trace", []),
        "retrieval_hits": result.get("retrieval_hits", []),
        "conflict": result.get("conflict"),
    }


def _inspector_content(
    *,
    snapshot: QuerySnapshot,
    original_question: str,
    standalone_question: str,
) -> InspectorContent:
    timeline, raw_trace = render_query_trace(snapshot.trace)
    retrieval_rounds = sum(event.stage == "retrieve" for event in snapshot.trace)
    query = render_query_details(
        snapshot.diagnostics,
        original_question=original_question,
        standalone_question=standalone_question,
        retrieval_rounds=retrieval_rounds,
    )
    return InspectorContent(
        sources=render_query_sources(snapshot.sources),
        retrieval=render_query_retrieval(snapshot.retrieval_hits),
        timeline=timeline,
        raw_trace=raw_trace,
        query=query,
    )


def submit_question(
    *,
    application: RAGApplication,
    message: str,
    history: ChatHistory,
    session_id: str,
) -> SubmitValues:
    """Adapt the existing decomposed chat callback to the Ask page components."""
    raw_values = application.chat(message, history, session_id)
    cleared = cast(str, raw_values[0])
    messages = cast(ChatHistory, raw_values[1])
    result = cast(PublicResult, raw_values[2])
    legacy_state = cast(str, raw_values[3])
    state = "unavailable" if message.strip() and not result else None
    snapshot = application._query_contract(result, state=state)
    answer_state = (
        legacy_state if not message.strip() else render_answer_state(snapshot.answer_state)
    )
    content = _inspector_content(
        snapshot=snapshot,
        original_question=message.strip(),
        standalone_question=str(result.get("standalone_query") or message.strip()),
    )
    safe_result = _public_result(
        application=application,
        messages=messages,
        result=result,
    )
    return (
        cleared,
        messages,
        safe_result,
        cast(ComponentUpdate, component_update(value=answer_state, visible=True)),
        content.sources,
        content.retrieval,
        content.timeline,
        content.raw_trace,
        content.query,
    )


def clear_conversation(
    *,
    application: RAGApplication,
    session_id: str,
) -> ClearValues:
    """Clear the graph checkpoint and rotate the browser session."""
    if application.rag_graph is not None:
        application.rag_graph.clear(session_id)
    empty = application._query_contract({}, answer="", state="completed")
    content = _inspector_content(
        snapshot=empty,
        original_question="",
        standalone_question="",
    )
    status = cast(
        ComponentUpdate,
        component_update(
            value=render_status("info", "No answer yet", "Conversation cleared."),
            visible=True,
        ),
    )
    return (
        [],
        {},
        status,
        content.sources,
        content.retrieval,
        content.timeline,
        content.raw_trace,
        content.query,
        str(uuid4()),
    )


def export_conversation(
    *,
    application: RAGApplication,
    messages: ChatHistory,
    result: PublicResult,
) -> ComponentUpdate:
    """Create a filtered public export and reveal its download action."""
    return cast(ComponentUpdate, application.export_chat_ui(messages, result))
