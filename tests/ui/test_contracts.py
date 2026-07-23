from dataclasses import FrozenInstanceError

import pytest

from modules.ui.contracts import (
    AttentionItem,
    CorpusSnapshot,
    DashboardSnapshot,
    EvaluationPageSnapshot,
    EvaluationSummary,
    IndexSnapshot,
    QueryDiagnostics,
    QuerySnapshot,
    RetrievalHitView,
    RuntimeSnapshot,
    SourceView,
    SystemCheck,
    TraceEventView,
)


def test_presentation_contracts_are_immutable_and_contain_plain_data() -> None:
    # Arrange
    check = SystemCheck("runtime", "Ollama connectivity", "ready", "Reachable")
    runtime = RuntimeSnapshot(
        state="ready",
        title="Ready for questions",
        detail="All local services are available.",
        chat_enabled=True,
        can_load_models=False,
        checks=(check,),
    )
    corpus = CorpusSnapshot(1, 2, 3, "ready")
    index = IndexSnapshot(0, 0, 0, 0, 0, "ready")
    evaluation = EvaluationSummary(
        result_path="/tmp/result.jsonl",
        split="development",
        systems=("dense", "bm25", "hybrid", "agentic"),
        case_count=10,
        result_kind="standard",
        created_at="2026-07-22T12:00:00Z",
    )
    dashboard = DashboardSnapshot(
        runtime=runtime,
        corpus=corpus,
        index=index,
        evaluation=evaluation,
        attention_items=(AttentionItem("runtime", "Review Ollama", "warning"),),
    )

    # Assert contracts stay immutable and compose without UI dependencies.
    assert dashboard.runtime.checks == (check,)
    assert dashboard.corpus.chunk_count == 3
    with pytest.raises(FrozenInstanceError):
        runtime.title = "Changed"  # type: ignore[misc]


def test_query_contract_keeps_public_observability_without_private_fields() -> None:
    # Arrange
    snapshot = QuerySnapshot(
        answer="Supported answer [C1].",
        answer_state="supported",
        sources=(SourceView("C1", "manual.pdf", 2, "Public excerpt"),),
        retrieval_hits=(
            RetrievalHitView("chunk-1", "manual.pdf", 2, 0.8, 4.0, 0.03, 0.77, ("part",)),
        ),
        trace=(TraceEventView("retrieve", "hybrid", 20, 12, 6, 1, 0, "completed", 12.5),),
        diagnostics=QueryDiagnostics(
            route="complex_search",
            retrieval_strategy="hybrid",
            subqueries=("part",),
            retry_count=1,
            evidence_state="supported",
            conflict_state="none",
            citation_validation="valid",
        ),
    )

    # Assert only public observability crosses the controller boundary.
    assert snapshot.sources[0].label == "C1"
    assert snapshot.retrieval_hits[0].semantic_score == 0.8
    assert "prompt" not in snapshot.__dataclass_fields__
    assert "reasoning" not in snapshot.__dataclass_fields__


def test_evaluation_page_contract_distinguishes_readiness_and_saved_result() -> None:
    snapshot = EvaluationPageSnapshot(
        state="saved_result",
        split="test",
        systems=("hybrid",),
        requires_index=True,
        requires_embeddings=True,
        requires_chat=False,
        problems=(),
        latest=None,
    )

    assert snapshot.state == "saved_result"
    assert snapshot.systems == ("hybrid",)
