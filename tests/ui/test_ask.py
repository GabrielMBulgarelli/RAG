from typing import Any, cast

from modules.ui import ask_presenters
from modules.ui.contracts import (
    CorpusSnapshot,
    QueryDiagnostics,
    QuerySnapshot,
    RetrievalHitView,
    RuntimeSnapshot,
    SourceView,
    TraceEventView,
)
from modules.ui.pages.ask_callbacks import refresh_ask_readiness
from modules.ui.shell import build_application


class AskApplication:
    rag_graph = None

    def runtime_snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            state="not_loaded",
            title="Application ready; AI not loaded",
            detail="Load AI models to enable document questions.",
            chat_enabled=False,
            can_load_models=True,
            checks=(),
        )

    def corpus_snapshot(self) -> CorpusSnapshot:
        return CorpusSnapshot(2, 5, 12, "ready")


class LoadedAskApplication(AskApplication):
    def runtime_snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            state="ready",
            title="AI models loaded",
            detail="Document questions are enabled.",
            chat_enabled=True,
            can_load_models=False,
            checks=(),
        )


def _query() -> QuerySnapshot:
    return QuerySnapshot(
        answer="Use the local index [C1].",
        answer_state="limited",
        sources=(
            SourceView(
                label="C1",
                filename="<guide.pdf>",
                page=2,
                excerpt="<Use local retrieval.>",
            ),
        ),
        retrieval_hits=(
            RetrievalHitView(
                chunk_id="<chunk-1>",
                filename="guide.pdf",
                page=2,
                semantic_score=0.8,
                sparse_score=1.2,
                fused_score=0.7,
                selection_score=0.9,
                matched_subqueries=("local retrieval",),
            ),
        ),
        trace=(
            TraceEventView(
                stage="<retrieve>",
                decision="hybrid",
                retrieved_count=4,
                fused_count=3,
                selected_count=1,
                retry_count=1,
                llm_calls=1,
                termination="limited",
                duration_ms=12.0,
            ),
        ),
        diagnostics=QueryDiagnostics(
            route="complex_search",
            retrieval_strategy="hybrid",
            subqueries=("local retrieval",),
            retry_count=1,
            evidence_state="limited",
            conflict_state="none",
            citation_validation="valid",
        ),
    )


def test_ask_page_is_a_two_column_workspace_with_required_controls() -> None:
    # When the routed application is constructed.
    config = build_application(cast(Any, AskApplication())).get_config_file()
    components = config["components"]
    values = [
        component["props"].get("value")
        for component in components
        if isinstance(component["props"].get("value"), str)
    ]
    labels = {component["props"].get("label") for component in components}
    ids = {component["props"].get("elem_id") for component in components}

    # Then the complete Ask workspace is exposed.
    assert "# Ask Documents" in values
    assert {"Indexed Documents", "Sources", "Retrieval", "Trace", "Query"} <= labels
    assert {
        "ask-model-status",
        "ask-header-actions",
        "run-evaluation",
        "ask-evaluation-status",
        "ask-workspace",
        "ask-conversation",
        "ask-inspector",
        "inspector-document-inventory",
        "inspector-selected-document",
        "inspector-indexing-errors",
    } <= ids
    assert "ask-mobile-inspector" not in ids
    assert "knowledge-filter" not in ids
    assert "ask-runtime-warning" not in ids
    assert {"Ask", "Clear", "Export", "Run evaluation"} <= set(values)
    assert "Download export" in labels
    model_status = next(
        component["props"]
        for component in components
        if component["props"].get("elem_id") == "ask-model-status"
    )
    assert "Model Not Loaded" in model_status["value"]
    question = next(
        component["props"]
        for component in components
        if component["props"].get("elem_id") == "ask-question"
    )
    assert question["lines"] == 3
    assert question["interactive"] is False


def test_ask_readiness_renders_not_loaded_model_status_and_preserves_controls() -> None:
    # When readiness is refreshed without loaded models.
    _, model_status, question, ask, load_models = refresh_ask_readiness(cast(Any, AskApplication()))
    model_status_value = cast(str, model_status.get("value", ""))

    # Then the unloaded badge and existing control states are returned.
    assert "ask-model-status--not-loaded" in model_status_value
    assert "Model Not Loaded" in model_status_value
    assert model_status.get("visible") is True
    assert question.get("interactive") is False
    assert ask.get("interactive") is False
    assert load_models.get("visible") is True
    assert load_models.get("interactive") is True


def test_ask_readiness_renders_loaded_model_status_and_preserves_controls() -> None:
    # When readiness is refreshed with loaded models.
    _, model_status, question, ask, load_models = refresh_ask_readiness(
        cast(Any, LoadedAskApplication())
    )
    model_status_value = cast(str, model_status.get("value", ""))

    # Then the loaded badge and existing control states are returned.
    assert "ask-model-status--loaded" in model_status_value
    assert "Model Loaded" in model_status_value
    assert "Model Not Loaded" not in model_status_value
    assert model_status.get("visible") is True
    assert question.get("interactive") is True
    assert ask.get("interactive") is True
    assert load_models.get("visible") is False
    assert load_models.get("interactive") is False


def test_ask_presenters_escape_public_observability_and_keep_raw_trace() -> None:
    # Given a query containing unsafe display values.
    snapshot = _query()

    # When public observability is rendered.
    answer = ask_presenters.render_answer_state(snapshot.answer_state)
    sources = ask_presenters.render_query_sources(snapshot.sources)
    retrieval = ask_presenters.render_query_retrieval(snapshot.retrieval_hits)
    timeline, raw_trace = ask_presenters.render_query_trace(snapshot.trace)
    query = ask_presenters.render_query_details(
        snapshot.diagnostics,
        original_question="<original>",
        standalone_question="<standalone>",
        retrieval_rounds=1,
    )

    # Then dynamic values are escaped without losing diagnostics.
    assert "Limited" in answer
    assert all(
        value in sources for value in ("C1", "&lt;guide.pdf&gt;", "Page 2", "Relevant", "Cited")
    )
    assert "<guide.pdf>" not in sources
    assert all(
        value in retrieval for value in ("&lt;chunk-1&gt;", "0.8000", "1.2000", "0.7000", "0.9000")
    )
    assert "&lt;Retrieve&gt;" in timeline
    assert all(value in timeline for value in ("Retrieved", "Fused", "Selected", "12 ms"))
    assert "<table" in raw_trace
    assert all(
        value in query
        for value in (
            "&lt;original&gt;",
            "&lt;standalone&gt;",
            "Complex Search",
            "Hybrid",
            "Retrieval rounds",
            "Valid",
        )
    )


def test_ask_submit_events_are_serialized_and_inspector_has_exact_tab_order() -> None:
    # When the routed application binds Ask events.
    interface = build_application(cast(Any, AskApplication()))
    config = interface.get_config_file()
    submit_functions = [
        function
        for function in interface.fns.values()
        if function.api_name in {"ask_with_button", "ask_with_enter"}
    ]

    # Then duplicate questions are serialized across both entry points.
    assert len(submit_functions) == 2
    assert all(function.concurrency_limit == 1 for function in submit_functions)
    assert all(
        "window.location.assign" not in (dependency.get("js") or "")
        for dependency in config.get("dependencies", [])
    )
    tab_labels = [
        component["props"].get("label")
        for component in config["components"]
        if component["type"] == "tabitem"
    ]
    assert tab_labels == ["Indexed Documents", "Sources", "Retrieval", "Trace", "Query"]
