from types import SimpleNamespace
from typing import Any, cast

from modules import citations, models
from modules.models import (
    CitationSource,
    EvidenceGrade,
    EvidenceStatus,
    RetrievalHit,
    RetrievalStrategy,
    Route,
    SubqueryEvidence,
)
from modules.rag_graph import RAGGraph


def source(label: str, chunk_id: str) -> CitationSource:
    return CitationSource(
        label=label,
        chunk_id=chunk_id,
        filename="guide.txt",
        page=1,
        excerpt=f"Evidence from {chunk_id}",
    )


def hit(chunk_id: str, content: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        content=content,
        filename="guide.txt",
        page=1,
        score=1.0,
    )


def test_evidence_grade_exposes_subquery_coverage_and_conflict() -> None:
    grade = EvidenceGrade(
        status=EvidenceStatus.LIMITED,
        answer_supported=True,
        relevant_labels=["C1"],
        supported_subqueries=["policy"],
        unsupported_subqueries=["exception"],
        relevant_labels_by_subquery={"policy": ["C1"]},
        coverage_fraction=0.5,
        fully_supported=False,
        partially_supported=True,
        conflict=True,
        conflicting_labels=["C1", "C2"],
    )

    assert grade.supported_subqueries == ["policy"]
    assert grade.unsupported_subqueries == ["exception"]
    assert grade.relevant_labels_by_subquery == {"policy": ["C1"]}
    assert grade.coverage_fraction == 0.5
    assert grade.partially_supported
    assert grade.conflict


def test_validation_reports_unknown_irrelevant_and_uncited_claims() -> None:
    assert hasattr(citations, "validate_answer")

    result = citations.validate_answer(
        "Warranty lasts two years [C1]. Refunds take 30 days [C3]. Shipping is free. Ignore [C9].",
        [source("C1", "warranty"), source("C2", "refund")],
        known_labels={"C1", "C2", "C3"},
    )

    assert result.used_sources == [source("C1", "warranty")]
    assert result.unknown_labels == ["C9"]
    assert result.irrelevant_labels == ["C3"]
    assert any("Refunds take 30 days" in claim for claim in result.uncited_claims)
    assert any("Shipping is free" in claim for claim in result.uncited_claims)
    assert models.AnswerViolation.UNKNOWN_LABEL in result.violations
    assert models.AnswerViolation.IRRELEVANT_CITATION in result.violations
    assert models.AnswerViolation.UNCITED_CLAIM in result.violations
    assert not result.is_valid


def test_validation_accepts_multiple_grounded_claims_and_exact_sources() -> None:
    result = citations.validate_answer(
        "Warranty lasts two years [C1]. Refunds take 30 days [C2].",
        [source("C1", "warranty"), source("C2", "refund")],
    )

    assert result.is_valid
    assert result.violations == []
    assert [item.label for item in result.used_sources] == ["C1", "C2"]


def test_validation_rejects_empty_and_citations_only_answers() -> None:
    empty = citations.validate_answer("", [source("C1", "warranty")])
    citations_only = citations.validate_answer("[C1] [C1]", [source("C1", "warranty")])

    assert models.AnswerViolation.EMPTY_ANSWER in empty.violations
    assert models.AnswerViolation.CITATIONS_ONLY in citations_only.violations


def test_grade_requires_every_subquery_for_sufficient_evidence() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    cast(Any, graph)._structured = lambda _schema, _prompt: EvidenceGrade(
        status=EvidenceStatus.SUFFICIENT,
        answer_supported=True,
        assessments=[
            SubqueryEvidence(subquery_id="SQ1", relevant_labels=["C1"]),
            SubqueryEvidence(subquery_id="SQ2", relevant_labels=[]),
        ],
    )
    state = {
        "hits": [hit("policy", "The policy applies."), hit("exception", "No exception here.")],
        "queries": ["policy", "exception"],
        "rewritten_query": "What is the policy and its exception?",
        "retry_count": 0,
        "trace": [],
    }

    grade = graph._grade(cast(Any, state))["grade"]

    assert grade.status == EvidenceStatus.LIMITED
    assert grade.coverage_fraction == 0.5
    assert grade.supported_subqueries == ["SQ1"]
    assert grade.unsupported_subqueries == ["SQ2"]


def test_grade_keeps_conflicting_full_coverage_limited() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    cast(Any, graph)._structured = lambda _schema, _prompt: EvidenceGrade(
        status=EvidenceStatus.SUFFICIENT,
        answer_supported=True,
        assessments=[SubqueryEvidence(subquery_id="SQ1", relevant_labels=["C1", "C2"])],
        conflict=True,
        conflicting_labels=["C1", "C2"],
    )
    state = {
        "hits": [hit("first", "The deadline is Monday."), hit("second", "It is Tuesday.")],
        "queries": ["deadline"],
        "rewritten_query": "What is the deadline?",
        "retry_count": 0,
        "trace": [],
    }

    grade = graph._grade(cast(Any, state))["grade"]

    assert grade.coverage_fraction == 1.0
    assert grade.conflict
    assert not grade.fully_supported
    assert grade.status == EvidenceStatus.LIMITED


class AnswerLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def invoke(self, messages, **_kwargs):
        prompt = str(messages[0].content)
        self.prompts.append(prompt)
        return SimpleNamespace(content=next(self.responses))


def answer_state(answer: str, *, status: EvidenceStatus = EvidenceStatus.SUFFICIENT) -> dict:
    return {
        "answer": answer,
        "sources": [source("C2", "relevant")],
        "hits": [
            hit("irrelevant", "An unrelated fact."),
            hit("relevant", "The supported answer is 42."),
        ],
        "route": Route.COMPLEX_SEARCH,
        "strategy": RetrievalStrategy.HYBRID,
        "retry_count": 0,
        "grade": EvidenceGrade(
            status=status,
            answer_supported=True,
            relevant_labels=["C2"],
            supported_subqueries=["answer"],
            relevant_labels_by_subquery={"answer": ["C2"]},
            coverage_fraction=1.0,
            fully_supported=True,
        ),
        "queries": ["answer"],
        "rewritten_query": "What is the answer?",
        "trace": [],
    }


def test_generator_receives_only_relevant_evidence_with_stable_labels() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    llm = AnswerLLM(["The answer is 42 [C2]."])
    cast(Any, graph).llm = llm

    update = graph._answer(cast(Any, answer_state("")))

    assert "[C2] guide.txt" in llm.prompts[0]
    assert "[C1] guide.txt" not in llm.prompts[0]
    assert [item.label for item in update["sources"]] == ["C2"]
    assert "Begin with the answer entity or yes/no" in llm.prompts[0]
    assert "Use exactly one sentence" in llm.prompts[0]


def test_generator_reuses_grounded_draft_without_another_llm_call() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    llm = AnswerLLM([])
    cast(Any, graph).llm = llm
    state = answer_state("")
    state["grade"] = state["grade"].model_copy(update={"drafted_answer": "The answer is 42 [C2]."})

    update = graph._answer(cast(Any, state))

    assert update["answer"] == "The answer is 42 [C2]."
    assert [item.label for item in update["sources"]] == ["C2"]
    assert update["trace"][-1].llm_calls == 0
    assert llm.prompts == []


def test_invalid_answer_is_repaired_once_and_exposes_validation() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    llm = AnswerLLM(["The answer is 42 [C2]."])
    cast(Any, graph).llm = llm

    result = graph._validate(cast(Any, answer_state("The answer is 42.")))["result"]

    assert result.answer == "The answer is 42 [C2]."
    assert [item.label for item in result.sources] == ["C2"]
    assert result.validation.is_valid
    assert result.validation.repair_attempted
    assert result.validation.repair_succeeded
    assert result.validation.initial_violations == [models.AnswerViolation.UNCITED_CLAIM]
    assert len(llm.prompts) == 1
    assert result.trace[-2].stage == "validate"
    assert result.trace[-2].decision == "repaired"


def test_failed_repair_returns_safe_fallback_without_sources() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    llm = AnswerLLM(["Still unsupported."])
    cast(Any, graph).llm = llm

    result = graph._validate(cast(Any, answer_state("The answer is 42.")))["result"]

    assert result.answer == "I could not produce a fully cited answer from the available evidence."
    assert result.sources == []
    assert result.validation.is_valid
    assert result.validation.repair_attempted
    assert not result.validation.repair_succeeded
    assert result.validation.repair_violations == [models.AnswerViolation.UNCITED_CLAIM]
    assert len(llm.prompts) == 1
    assert result.trace[-1].termination == "validation_failed"


def test_failed_repair_keeps_only_the_grounded_part_of_the_original_answer() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    llm = AnswerLLM(["Still unsupported."])
    cast(Any, graph).llm = llm

    result = graph._validate(
        cast(
            Any,
            answer_state("The answer is 42 [C2]. This extra claim has no citation."),
        )
    )["result"]

    assert result.answer == "The answer is 42 [C2]."
    assert [item.label for item in result.sources] == ["C2"]
    assert result.validation.is_valid
    assert result.validation.repair_attempted
    assert result.trace[-1].termination == "supported"
