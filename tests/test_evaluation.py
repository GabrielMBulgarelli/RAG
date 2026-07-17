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
    MetricObservation,
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


def assert_observation(
    metrics: dict[str, MetricObservation],
    name: str,
    *,
    value: float | None,
    status: str,
    sample_count: int,
) -> None:
    observation = metrics[name]
    assert observation.value == pytest.approx(value) if value is not None else observation.value is None
    assert observation.status == status
    assert observation.sample_count == sample_count


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
    assert_observation(
        metrics, "route_accuracy", value=0.5, status="measured", sample_count=2
    )
    assert_observation(
        metrics, "strategy_accuracy", value=1.0, status="measured", sample_count=2
    )


def test_retry_precision_and_recall_edge_cases() -> None:
    negatives = [case(id="n", expected_retry=False)]
    metrics = aggregate_metrics(negatives, [result(case_id="n")])
    assert_observation(
        metrics, "retry_precision", value=None, status="no_eligible_cases", sample_count=0
    )
    assert_observation(
        metrics, "retry_recall", value=None, status="no_eligible_cases", sample_count=0
    )

    cases = [case(id="p", expected_retry=True), case(id="n", expected_retry=False)]
    metrics = aggregate_metrics(
        cases,
        [result(case_id="p", retry_count=1), result(case_id="n", retry_count=1)],
    )
    assert_observation(
        metrics, "retry_precision", value=0.5, status="measured", sample_count=2
    )
    assert_observation(
        metrics, "retry_recall", value=1.0, status="measured", sample_count=1
    )

    missed = aggregate_metrics([case(expected_retry=True)], [result()])
    assert_observation(
        missed, "retry_precision", value=None, status="no_eligible_cases", sample_count=0
    )
    assert_observation(
        missed, "retry_recall", value=0.0, status="measured", sample_count=1
    )


def test_citation_metrics() -> None:
    assert citation_precision(["a", "unknown"], ["a", "b"]) == 0.5
    assert citation_precision([], ["a"]) is None
    assert gold_citation_coverage(["a"], ["a", "b"]) == 0.5
    assert gold_citation_coverage([], []) is None


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
    assert_observation(
        metrics, "abstention_accuracy", value=1.0, status="measured", sample_count=2
    )
    assert_observation(
        metrics,
        "unanswerable_abstention_recall",
        value=1.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(
        metrics,
        "answerable_response_rate",
        value=1.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(
        metrics, "conflict_recall", value=1.0, status="measured", sample_count=1
    )
    assert_observation(
        metrics,
        "conflict_false_positive_rate",
        value=0.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(
        metrics, "termination_rate", value=0.5, status="measured", sample_count=2
    )
    assert "conflict_accuracy" not in metrics


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
    output = write_experiment(
        tmp_path,
        config,
        [result()],
        {
            "agentic": {
                "recall_at_5": MetricObservation(
                    value=1.0,
                    status="measured",
                    sample_count=1,
                    note="Cases with expected chunk evidence.",
                )
            }
        },
    )

    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["configuration"]["systems"] == ["dense", "agentic"]
    assert summary["metrics"]["agentic"]["recall_at_5"] == {
        "value": 1.0,
        "status": "measured",
        "sample_count": 1,
        "note": "Cases with expected chunk evidence.",
    }
    assert (output / "cases.jsonl").read_text().strip()
    markdown = (output / "summary.md").read_text()
    assert "agentic" in markdown
    assert "1.000000 · n=1" in markdown
    assert "Cases with expected chunk evidence." in markdown


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
    assert "chunk_recall_at_5" not in metrics
    assert_observation(
        metrics, "recall_at_5", value=0.5, status="measured", sample_count=1
    )
    assert_observation(
        metrics, "document_recall_at_5", value=0.5, status="measured", sample_count=1
    )
    assert_observation(
        metrics,
        "normalized_answer_exact_match",
        value=1.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(
        metrics, "answer_token_f1", value=1.0, status="measured", sample_count=1
    )
    assert normalized_exact_match("The Answer!", "the answer") == 1.0
    assert token_f1("alpha beta", "alpha gamma") == 0.5


def test_retrieval_only_response_metrics_are_not_applicable() -> None:
    metrics = aggregate_metrics([case()], [result(system="dense", route=None, strategy=None)])

    assert_observation(
        metrics, "recall_at_5", value=0.5, status="measured", sample_count=1
    )
    for name in (
        "citation_precision",
        "gold_evidence_citation_coverage",
        "abstention_accuracy",
        "unanswerable_abstention_recall",
        "answerable_response_rate",
        "conflict_recall",
        "conflict_false_positive_rate",
        "normalized_answer_exact_match",
        "answer_token_f1",
    ):
        assert_observation(
            metrics, name, value=None, status="not_applicable", sample_count=0
        )


def test_conditioned_metrics_distinguish_empty_denominators_from_measured_zero() -> None:
    metrics = aggregate_metrics(
        [case(expected_answer="expected", expected_conflict=False)],
        [result(cited_chunk_ids=["unknown"], answer="different")],
    )

    assert_observation(
        metrics, "citation_precision", value=0.0, status="measured", sample_count=1
    )
    assert_observation(
        metrics,
        "gold_evidence_citation_coverage",
        value=0.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(
        metrics, "answer_token_f1", value=0.0, status="measured", sample_count=1
    )
    assert_observation(
        metrics, "conflict_recall", value=None, status="no_eligible_cases", sample_count=0
    )
    assert_observation(
        metrics,
        "unanswerable_abstention_recall",
        value=None,
        status="no_eligible_cases",
        sample_count=0,
    )


def test_no_emitted_citations_are_no_eligible_cases() -> None:
    metrics = aggregate_metrics([case()], [result(cited_chunk_ids=[])])
    assert_observation(
        metrics, "citation_precision", value=None, status="no_eligible_cases", sample_count=0
    )


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
