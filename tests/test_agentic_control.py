from __future__ import annotations

from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any, cast

from modules import rag_graph
from modules.models import (
    EvidenceGrade,
    EvidenceStatus,
    RetrievalBatch,
    RetrievalHit,
    RetrievalStrategy,
    Route,
    RouteDecision,
)
from modules.rag_graph import RAGGraph


def _hit(chunk_id: str, score: float, query: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        content=f"Evidence for {chunk_id}",
        filename="guide.txt",
        page=1,
        score=score,
        fused_score=score,
        subqueries=[query],
    )


def test_deterministic_routes_avoid_the_llm_for_clear_requests() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    graph.vector_db = cast(Any, SimpleNamespace(document_names=lambda: ["guide.txt"]))
    calls: list[str] = []
    graph._structured = lambda _schema, prompt: calls.append(prompt)  # type: ignore[method-assign]

    cases = [
        ("Which documents are indexed?", Route.CATALOG, RetrievalStrategy.NONE),
        ("What is the annual leave policy?", Route.SIMPLE_SEARCH, RetrievalStrategy.SEMANTIC),
        (
            "Compare the leave policies in the employee handbook and contractor guide.",
            Route.COMPLEX_SEARCH,
            RetrievalStrategy.HYBRID,
        ),
    ]
    for query, route, strategy in cases:
        update = graph._route(cast(Any, {"rewritten_query": query, "trace": []}))
        assert update["route"] == route
        assert update["strategy"] == strategy
        assert update["trace"][-1].llm_calls == 0

    assert calls == []


def test_uncertain_route_uses_one_llm_call() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    graph.vector_db = cast(Any, SimpleNamespace(document_names=lambda: ["guide.txt"]))
    calls: list[str] = []

    def decide(_schema: type, prompt: str) -> RouteDecision:
        calls.append(prompt)
        return RouteDecision(route=Route.SIMPLE_SEARCH, strategy=RetrievalStrategy.HYBRID)

    graph._structured = decide  # type: ignore[method-assign]
    update = graph._route(cast(Any, {"rewritten_query": "Explain the policy.", "trace": []}))

    assert update["route"] == Route.SIMPLE_SEARCH
    assert update["trace"][-1].llm_calls == 1
    assert len(calls) == 1


def test_independent_subqueries_run_concurrently_with_a_four_worker_cap() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    lock = Lock()
    active = 0
    maximum_active = 0

    def search(query: str, **_kwargs: object) -> RetrievalBatch:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        sleep(0.04)
        with lock:
            active -= 1
        return RetrievalBatch(
            hits=[_hit(query, 1.0, query)],
            retrieved_count=1,
            fused_count=1,
            selected_count=1,
        )

    graph.retriever = cast(Any, SimpleNamespace(search=search))
    state = {
        "queries": [f"query-{index}" for index in range(5)],
        "required_queries": [f"query-{index}" for index in range(5)],
        "rewritten_query": "complex question",
        "strategy": RetrievalStrategy.HYBRID,
        "retry_count": 0,
        "trace": [],
        "filters": {},
    }

    update = graph._retrieve(cast(Any, state))

    assert 1 < maximum_active <= 4
    assert [hit.chunk_id for hit in update["hits"]] == [
        "query-0",
        "query-1",
        "query-2",
        "query-3",
        "query-4",
    ]


def test_retry_targets_only_unsupported_subqueries_and_preserves_valid_hits() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    first = _hit("supported", 0.01, "policy")
    state = {
        "queries": ["policy", "exception"],
        "required_queries": ["policy", "exception"],
        "hits": [first, _hit("unused", 0.9, "exception")],
        "grade": EvidenceGrade(
            status=EvidenceStatus.LIMITED,
            relevant_labels=["C1"],
            supported_subqueries=["policy"],
            unsupported_subqueries=["exception"],
            relevant_labels_by_subquery={"policy": ["C1"]},
        ),
        "retry_count": 0,
        "trace": [],
    }

    update = graph._refine(cast(Any, state))

    assert update["queries"] == ["exception"]
    assert [hit.chunk_id for hit in update["preserved_hits"]] == ["supported"]
    assert update["retry_count"] == 1
    assert update["trace"][-1].llm_calls == 0


def test_retry_merges_new_results_without_evicting_preserved_evidence(
    monkeypatch,
) -> None:
    graph = RAGGraph.__new__(RAGGraph)
    preserved = _hit("supported", 0.001, "policy")
    graph.retriever = cast(
        Any,
        SimpleNamespace(
            search=lambda query, **_kwargs: RetrievalBatch(
                hits=[_hit("new-a", 1.0, query), _hit("new-b", 0.9, query)],
                retrieved_count=2,
                fused_count=2,
                selected_count=2,
            )
        ),
    )
    monkeypatch.setattr(rag_graph.config, "max_context_chunks", 2)
    state = {
        "queries": ["exception"],
        "required_queries": ["policy", "exception"],
        "preserved_hits": [preserved],
        "rewritten_query": "policy and exception",
        "strategy": RetrievalStrategy.HYBRID,
        "retry_count": 1,
        "trace": [],
        "filters": {},
    }

    update = graph._retrieve(cast(Any, state))

    assert "supported" in {hit.chunk_id for hit in update["hits"]}
    assert len(update["hits"]) == 2


def test_trace_exposes_stage_llm_calls_and_bounded_retry() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    state = {
        "queries": ["missing evidence"],
        "required_queries": ["missing evidence"],
        "hits": [],
        "grade": EvidenceGrade(
            status=EvidenceStatus.INSUFFICIENT,
            unsupported_subqueries=["missing evidence"],
        ),
        "retry_count": 0,
        "trace": [],
    }

    retry = graph._refine(cast(Any, state))

    assert retry["retry_count"] == 1
    assert retry["trace"][-1].duration_ms >= 0
    assert retry["trace"][-1].llm_calls == 0
    assert rag_graph.decide_after_grading(EvidenceStatus.INSUFFICIENT, 1) == "abstain"
