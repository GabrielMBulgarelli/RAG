import json
import urllib.error
from pathlib import Path
from typing import cast

import pytest

from modules import evaluation
from modules.evaluation import (
    BenchmarkEvidence,
    CaseResult,
    CountingModel,
    EvaluationCase,
    ExperimentConfig,
    aggregate_metrics,
    citation_precision,
    failure_labels,
    filter_cases,
    gold_citation_coverage,
    mrr_at_k,
    ndcg_at_k,
    normalized_exact_match,
    p95,
    recall_at_k,
    run_agentic_case,
    token_f1,
    write_experiment,
)


def case(**updates: object) -> EvaluationCase:
    values: dict[str, object] = {
        "id": "case-1",
        "split": "development",
        "category": "search",
        "question": "question",
        "answerable": True,
        "relevant_chunk_ids": ["a", "b"],
        "expected_route": "simple_search",
        "expected_strategy": "semantic",
        "expected_retry": False,
        "expected_conflict": False,
    }
    values.update(updates)
    return EvaluationCase.model_validate(values)


def result(**updates: object) -> CaseResult:
    values: dict[str, object] = {
        "case_id": "case-1",
        "system": "agentic",
        "retrieved_chunk_ids": ["a", "x"],
        "cited_chunk_ids": ["a"],
        "route": "simple_search",
        "strategy": "semantic",
        "retry_count": 0,
        "conflict_detected": False,
        "terminated": True,
        "abstained": False,
        "latency_seconds": 0.1,
        "llm_calls": 2,
        "retrieval_rounds": 1,
    }
    values.update(updates)
    return CaseResult.model_validate(values)


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [([], 0.0), (["a", "x"], 0.5), (["b", "a"], 1.0)],
)
def test_recall_at_five(retrieved: list[str], expected: float) -> None:
    assert recall_at_k(retrieved, {"a", "b"}, 5) == expected


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [(["a", "x"], 1.0), (["x", "y", "a"], 1 / 3), (["x"], 0.0)],
)
def test_mrr_at_five(retrieved: list[str], expected: float) -> None:
    assert mrr_at_k(retrieved, {"a"}, 5) == pytest.approx(expected)


def test_ndcg_at_five_uses_binary_relevance() -> None:
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    expected = (1 / 1.5849625007) / (1 + 1 / 1.5849625007)
    assert ndcg_at_k(["x", "a"], {"a", "b"}, 5) == pytest.approx(expected)
    assert ndcg_at_k(["x"], {"a"}, 5) == 0.0


def test_p95_handles_small_and_normal_samples() -> None:
    assert p95([]) == 0.0
    assert p95([2.0]) == 2.0
    assert p95(list(map(float, range(1, 101)))) == pytest.approx(95.05)


def test_agent_accuracy_denominators_exclude_non_agentic_results() -> None:
    cases = [case(id="a"), case(id="b", expected_strategy="hybrid")]
    results = [
        result(case_id="a"),
        result(case_id="b", route="complex_search", strategy="hybrid"),
        result(case_id="a", system="dense", route=None, strategy=None),
    ]
    metrics = aggregate_metrics(cases, results)
    assert metrics["route_accuracy"] == 0.5
    assert metrics["strategy_accuracy"] == 1.0


def test_retry_precision_and_recall_edge_cases() -> None:
    negatives = [case(id="n", expected_retry=False)]
    assert aggregate_metrics(negatives, [result(case_id="n")])["retry_precision"] == 1.0
    assert aggregate_metrics(negatives, [result(case_id="n")])["retry_recall"] == 1.0

    cases = [case(id="p", expected_retry=True), case(id="n", expected_retry=False)]
    metrics = aggregate_metrics(
        cases,
        [result(case_id="p", retry_count=1), result(case_id="n", retry_count=1)],
    )
    assert metrics["retry_precision"] == 0.5
    assert metrics["retry_recall"] == 1.0

    missed = aggregate_metrics([case(expected_retry=True)], [result()])
    assert missed["retry_precision"] == 0.0
    assert missed["retry_recall"] == 0.0


def test_citation_metrics() -> None:
    assert citation_precision(["a", "unknown"], ["a", "b"]) == 0.5
    assert citation_precision([], ["a"]) == 1.0
    assert gold_citation_coverage(["a"], ["a", "b"]) == 0.5
    assert gold_citation_coverage([], []) == 1.0


def test_abstention_conflict_and_termination_metrics() -> None:
    cases = [
        case(id="answerable", answerable=True),
        case(id="no-answer", answerable=False, relevant_chunk_ids=[], expected_conflict=True),
    ]
    metrics = aggregate_metrics(
        cases,
        [
            result(case_id="answerable"),
            result(
                case_id="no-answer",
                abstained=True,
                conflict_detected=True,
                terminated=False,
            ),
        ],
    )
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["conflict_accuracy"] == 1.0
    assert metrics["termination_rate"] == 0.5


def test_split_filtering() -> None:
    cases = [case(id="dev"), case(id="held", split="test")]
    assert [item.id for item in filter_cases(cases, "test")] == ["held"]


def test_checked_in_dataset_has_seventy_thirty_split() -> None:
    dataset = Path(__file__).parents[1] / "evals" / "mvp_cases.jsonl"
    cases = evaluation.load_cases(dataset)
    assert len(filter_cases(cases, "development")) == 7
    assert len(filter_cases(cases, "test")) == 3


def test_missing_ollama_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(evaluation.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="Start Ollama"):
        evaluation._require_ollama()


class FakeModel:
    structured_kwargs: dict[str, object] = {}

    def invoke(self, value: object, **kwargs: object) -> str:
        return f"response:{value}"

    def with_structured_output(self, schema: object, **kwargs: object) -> "FakeModel":
        self.structured_kwargs = kwargs
        return self


def test_llm_calls_are_counted_at_shared_boundary() -> None:
    model = FakeModel()
    counted = CountingModel(model)
    assert counted.invoke("one") == "response:one"
    assert counted.with_structured_output(dict, method="json_schema").invoke("two") == (
        "response:two"
    )
    assert counted.calls == 2
    assert model.structured_kwargs == {"method": "json_schema"}


def test_experiment_configuration_serializes(tmp_path: Path) -> None:
    config = ExperimentConfig(
        run_id="run",
        timestamp="2026-01-02T03:04:05Z",
        git_commit="abc123",
        dataset_hash="sha256",
        evaluated_split="test",
        systems=["dense", "agentic"],
        chat_model="chat",
        embedding_model="embed",
        chunk_size=700,
        chunk_overlap=100,
        retrieval_limit=5,
        semantic_candidates=10,
        sparse_candidates=10,
        retry_limit=1,
        subquery_limit=4,
    )
    output = write_experiment(tmp_path, config, [result()], {"agentic": {"recall_at_5": 1.0}})

    summary = json.loads((output / "summary.json").read_text())
    assert summary["configuration"]["systems"] == ["dense", "agentic"]
    assert summary["metrics"]["agentic"]["recall_at_5"] == 1.0
    assert (output / "cases.jsonl").read_text().strip()
    assert "agentic" in (output / "summary.md").read_text()


def test_failure_labels_are_deterministic_and_composable() -> None:
    labels = failure_labels(
        case(expected_retry=True, expected_conflict=True),
        result(
            route="complex_search",
            strategy="hybrid",
            retrieved_chunk_ids=["x"],
            cited_chunk_ids=["unknown"],
            retry_count=0,
            conflict_detected=False,
            terminated=False,
            abstained=True,
        ),
    )
    assert labels == [
        "route_error",
        "strategy_error",
        "retrieval_miss",
        "invalid_citation",
        "citation_coverage_miss",
        "over_abstention",
        "retry_error",
        "conflict_miss",
        "non_termination",
    ]


def test_runtime_and_failed_abstention_labels() -> None:
    assert failure_labels(
        case(answerable=False, relevant_chunk_ids=[]),
        result(abstained=False, runtime_error="boom"),
    ) == [
        "failed_abstention",
        "runtime_error",
    ]


def test_agentic_runtime_error_is_recorded_without_aborting() -> None:
    class FailingGraph:
        def process_query(self, question: str, session_id: str) -> dict[str, object]:
            raise AttributeError("structured output was empty")

    class FakeModel:
        calls = 3

    measured = run_agentic_case(
        case(),
        cast(evaluation.RAGGraph, FailingGraph()),
        cast(CountingModel, FakeModel()),
    )

    assert measured.runtime_error == "AttributeError: structured output was empty"
    assert measured.terminated is False
    assert "runtime_error" in measured.failure_labels
    assert "non_termination" in measured.failure_labels


def test_document_and_answer_metrics() -> None:
    benchmark = case(
        relevant_chunk_ids=["c1", "c2"],
        relevant_document_ids=["d1", "d2"],
        expected_answer="Sam Bankman-Fried",
    )
    measured = result(
        retrieved_chunk_ids=["c1", "x"],
        retrieved_document_ids=["d1", "other"],
        answer="Sam Bankman Fried",
    )
    metrics = aggregate_metrics([benchmark], [measured])
    assert metrics["chunk_recall_at_5"] == 0.5
    assert metrics["document_recall_at_5"] == 0.5
    assert metrics["normalized_answer_exact_match"] == 1.0
    assert metrics["answer_token_f1"] == 1.0
    assert normalized_exact_match("The Answer!", "the answer") == 1.0
    assert token_f1("alpha beta", "alpha gamma") == 0.5


def test_benchmark_evidence_schema_preserves_stable_source() -> None:
    evidence = BenchmarkEvidence(
        benchmark_document_id="doc-abc",
        source="The Verge",
        title="A title",
        url="https://example.test/story",
        evidence_text="Exact benchmark fact.",
    )
    benchmark = case(gold_evidence=[evidence], relevant_chunk_ids=[])
    assert benchmark.gold_evidence[0].benchmark_document_id == "doc-abc"
