from __future__ import annotations

from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any, cast

from modules import rag_graph
from modules.models import (
    EvidenceDecision,
    EvidenceGrade,
    EvidenceStatus,
    QueryRefinement,
    RetrievalBatch,
    RetrievalHit,
    RetrievalStrategy,
    Route,
    RouteDecision,
    SubqueryEvidence,
    SubquerySpec,
)
from modules.rag_graph import RAGGraph, make_chat_model


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


def test_chat_model_bounds_generation_and_http_wait(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat_ollama(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(rag_graph, "ChatOllama", fake_chat_ollama)

    model = make_chat_model("test-model")

    assert model is not None
    assert captured["num_predict"] == 512
    assert captured["client_kwargs"] == {"timeout": 60.0}


def test_evidence_grade_requires_an_explicit_answer_support_decision() -> None:
    assert "answer_supported" in EvidenceGrade.model_json_schema()["required"]


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


def test_complex_search_preserves_the_full_question_without_an_llm_call() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    graph._structured = lambda *_args: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("complex retrieval must not rewrite away query constraints")
    )
    question = (
        "What company was scrutinized for its ad-free subscription and also for "
        "suppressing Palestinian voices?"
    )

    update = graph._decompose(cast(Any, {"rewritten_query": question, "trace": []}))

    assert update["queries"] == [SubquerySpec(id="SQ1", text=question)]
    assert update["required_queries"] == [SubquerySpec(id="SQ1", text=question)]
    assert update["trace"][-1].llm_calls == 0


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


def test_retrieval_preserves_one_candidate_for_each_subquery(monkeypatch) -> None:
    graph = RAGGraph.__new__(RAGGraph)

    def search(query: str, **_kwargs: object) -> RetrievalBatch:
        hits = (
            [_hit("dominant-a", 1.0, query), _hit("dominant-b", 0.9, query)]
            if query == "first"
            else [_hit("only-second", 0.01, query)]
        )
        return RetrievalBatch(
            hits=hits,
            retrieved_count=len(hits),
            fused_count=len(hits),
            selected_count=len(hits),
        )

    graph.retriever = cast(Any, SimpleNamespace(search=search))
    monkeypatch.setattr(rag_graph.config, "max_context_chunks", 2)
    state = {
        "queries": ["first", "second"],
        "required_queries": ["first", "second"],
        "rewritten_query": "combined",
        "strategy": RetrievalStrategy.HYBRID,
        "retry_count": 0,
        "trace": [],
        "filters": {},
    }

    update = graph._retrieve(cast(Any, state))

    assert {hit.chunk_id for hit in update["hits"]} == {"dominant-a", "only-second"}


def test_retry_targets_only_unsupported_subqueries_and_preserves_valid_hits() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    first = _hit("supported", 0.01, "policy")
    graph._structured = lambda _schema, _prompt: QueryRefinement(  # type: ignore[method-assign]
        rewrites={"SQ2": "policy exception eligibility requirements"}
    )
    state = {
        "queries": [
            SubquerySpec(id="SQ1", text="policy"),
            SubquerySpec(id="SQ2", text="exception"),
        ],
        "required_queries": [
            SubquerySpec(id="SQ1", text="policy"),
            SubquerySpec(id="SQ2", text="exception"),
        ],
        "hits": [first, _hit("unused", 0.9, "exception")],
        "grade": EvidenceGrade(
            status=EvidenceStatus.LIMITED,
            answer_supported=True,
            relevant_labels=["C1"],
            supported_subqueries=["SQ1"],
            unsupported_subqueries=["SQ2"],
            relevant_labels_by_subquery={"SQ1": ["C1"]},
        ),
        "retry_count": 0,
        "trace": [],
    }

    update = graph._refine(cast(Any, state))

    assert update["queries"] == [
        SubquerySpec(id="SQ2", text="policy exception eligibility requirements")
    ]
    assert [hit.chunk_id for hit in update["preserved_hits"]] == ["supported"]
    assert update["retry_count"] == 1
    assert update["trace"][-1].llm_calls == 1


def test_grade_uses_stable_subquery_ids_instead_of_matching_query_prose() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    requested_schemas: list[str] = []

    def grade(schema: type, _prompt: str) -> EvidenceDecision:
        requested_schemas.append(schema.__name__)
        return EvidenceDecision(
            answer_supported=True,
            drafted_answer="The founder was born in London [C1] [C2].",
            assessments=[
                SubqueryEvidence(subquery_id="SQ1", relevant_labels=["C1"]),
                SubqueryEvidence(subquery_id="SQ2", relevant_labels=["C2"]),
            ],
        )

    graph._structured = grade  # type: ignore[method-assign]
    state = {
        "required_queries": [
            SubquerySpec(id="SQ1", text="Who founded the company?"),
            SubquerySpec(id="SQ2", text="Where was that person born?"),
        ],
        "queries": [],
        "rewritten_query": "Who founded the company and where were they born?",
        "hits": [_hit("founder", 1.0, "founder"), _hit("birthplace", 0.9, "birthplace")],
        "retry_count": 0,
        "trace": [],
    }

    grade = graph._grade(cast(Any, state))["grade"]

    assert grade.supported_subqueries == ["SQ1", "SQ2"]
    assert grade.unsupported_subqueries == []
    assert grade.relevant_labels_by_subquery == {"SQ1": ["C1"], "SQ2": ["C2"]}
    assert grade.status == EvidenceStatus.SUFFICIENT
    assert grade.drafted_answer == "The founder was born in London [C1] [C2]."
    assert requested_schemas == ["EvidenceDecision"]


def test_grade_prompt_accepts_explicit_entity_linking_across_sources() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    prompts: list[str] = []

    def grade(schema: type, prompt: str) -> EvidenceDecision:
        assert schema is EvidenceDecision
        prompts.append(prompt)
        return EvidenceDecision(
            answer_supported=True,
            assessments=[
                SubqueryEvidence(
                    subquery_id="SQ1",
                    relevant_labels=["C1", "C2"],
                )
            ],
        )

    graph._structured = grade  # type: ignore[method-assign]
    state = {
        "required_queries": [
            SubquerySpec(id="SQ1", text="Identify the company named in both reports."),
        ],
        "queries": [],
        "rewritten_query": "Which company is described in both reports?",
        "hits": [
            _hit("complaint", 1.0, "A consumer group filed a complaint against Meta."),
            _hit(
                "moderation",
                0.9,
                "Meta was accused of suppressing Palestinian voices.",
            ),
        ],
        "retry_count": 0,
        "trace": [],
    }

    graph._grade(cast(Any, state))

    grading_prompt = " ".join(prompts[0].lower().split())
    assert (
        "a shared entity explicitly named across independent evidence items directly "
        "supports an entity-linking answer"
    ) in grading_prompt
    assert "shared names, topics, or partial clues are not enough" not in grading_prompt


def test_grade_rejects_partial_clues_that_do_not_answer_the_question() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    graph._structured = lambda _schema, _prompt: EvidenceGrade(  # type: ignore[method-assign]
        status=EvidenceStatus.LIMITED,
        answer_supported=False,
        assessments=[SubqueryEvidence(subquery_id="SQ1", relevant_labels=["C1"])],
    )
    state = {
        "required_queries": [
            SubquerySpec(id="SQ1", text="Which company was investigated?"),
            SubquerySpec(id="SQ2", text="What is that company's stock symbol?"),
        ],
        "queries": [],
        "rewritten_query": "What single-letter stock symbol belongs to the investigated company?",
        "hits": [_hit("investigation", 1.0, "company")],
        "retry_count": 0,
        "trace": [],
    }

    grade = graph._grade(cast(Any, state))["grade"]

    assert grade.status == EvidenceStatus.INSUFFICIENT
    assert not grade.answer_supported


def test_evidence_present_failure_does_not_enter_expensive_retry() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    state = {
        "grade": EvidenceGrade(
            status=EvidenceStatus.INSUFFICIENT,
            answer_supported=False,
        ),
        "hits": [_hit("partial", 1.0, "question")],
        "retry_count": 0,
    }

    assert graph._after_grade(cast(Any, state)) == "abstain"


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
    graph._structured = lambda _schema, _prompt: QueryRefinement(  # type: ignore[method-assign]
        rewrites={"SQ1": "missing evidence"}
    )
    state = {
        "queries": [SubquerySpec(id="SQ1", text="missing evidence")],
        "required_queries": [SubquerySpec(id="SQ1", text="missing evidence")],
        "hits": [],
        "grade": EvidenceGrade(
            status=EvidenceStatus.INSUFFICIENT,
            answer_supported=False,
            unsupported_subqueries=["SQ1"],
        ),
        "retry_count": 0,
        "trace": [],
    }

    retry = graph._refine(cast(Any, state))

    assert retry["retry_count"] == 1
    assert retry["queries"] == []
    assert retry["termination_hint"] == "retry_noop"
    assert retry["trace"][-1].duration_ms >= 0
    assert retry["trace"][-1].llm_calls == 1
    assert rag_graph.decide_after_grading(EvidenceStatus.INSUFFICIENT, 1) == "abstain"
