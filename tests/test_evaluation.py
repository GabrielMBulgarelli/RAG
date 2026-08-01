import io
import json
import sys
import time
import urllib.error
from pathlib import Path
from typing import cast

import pytest

from modules import evaluation, evaluation_models
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
    map_retrieved_evidence,
    mrr_at_k,
    ndcg_at_k,
    normalized_exact_match,
    p95,
    recall_at_k,
    run_agentic_case,
    run_fixed_rag_case,
    token_f1,
    write_experiment,
)
from modules.evaluation_metrics import aggregate_metrics as decomposed_aggregate_metrics
from modules.evaluation_models import EvaluationCase as DecomposedEvaluationCase
from modules.evaluation_reporting import write_experiment as decomposed_write_experiment

EXPECTED_SYSTEMS = (
    "dense",
    "bm25",
    "hybrid",
    "dense-rag",
    "bm25-rag",
    "hybrid-rag",
    "full-rag",
)


def test_evaluation_facade_preserves_decomposed_public_api() -> None:
    assert EvaluationCase is DecomposedEvaluationCase
    assert aggregate_metrics is decomposed_aggregate_metrics
    assert write_experiment is decomposed_write_experiment


def test_retrieved_evidence_mapping_prefers_excerpt_and_normalizes_content() -> None:
    long_excerpt = "  Preferred\n\ttext  " + (" word" * 80)
    bounded_excerpt = " ".join(long_excerpt.split())[:300]

    evidence = map_retrieved_evidence(
        [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "filename": "guide.pdf",
                "page": 4,
                "excerpt": long_excerpt,
                "content": "Ignored content",
            },
            {
                "chunk_id": "chunk-2",
                "document_id": "doc-2",
                "filename": "manual.pdf",
                "page": 9,
                "content": "  Content-only\n evidence\t remains readable.  ",
            },
        ]
    )

    assert evidence[0] == {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "filename": "guide.pdf",
        "page": 4,
        "excerpt": bounded_excerpt,
    }
    assert len(bounded_excerpt) == 300
    assert evidence[1] == {
        "chunk_id": "chunk-2",
        "document_id": "doc-2",
        "filename": "manual.pdf",
        "page": 9,
        "excerpt": "Content-only evidence remains readable.",
    }


def test_full_rag_benchmark_has_exact_retrieval_fixed_rag_and_full_rag_system_order() -> None:
    # Given / When
    systems = evaluation_models.SYSTEMS

    # Then
    assert systems == EXPECTED_SYSTEMS
    assert evaluation_models.RETRIEVAL_SYSTEMS == ("dense", "bm25", "hybrid")
    assert evaluation_models.FIXED_RAG_SYSTEMS == ("dense-rag", "bm25-rag", "hybrid-rag")
    assert evaluation_models.ANSWER_SYSTEMS == (*evaluation_models.FIXED_RAG_SYSTEMS, "full-rag")
    assert evaluation_models.FULL_RAG_SYSTEM == "full-rag"


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
        "system": "full-rag",
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
    assert (
        observation.value == pytest.approx(value)
        if value is not None
        else observation.value is None
    )
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
    assert_observation(metrics, "route_accuracy", value=0.5, status="measured", sample_count=2)
    assert_observation(metrics, "strategy_accuracy", value=1.0, status="measured", sample_count=2)


def test_fixed_rag_measures_answers_but_only_full_rag_measures_workflow_decisions() -> None:
    # Arrange
    benchmark = case(expected_answer="expected answer")
    fixed = aggregate_metrics(
        [benchmark],
        [result(system="dense-rag", answer="expected answer", route=None, strategy=None)],
        system="dense-rag",
    )
    full = aggregate_metrics(
        [benchmark],
        [result(system="full-rag", answer="expected answer")],
        system="full-rag",
    )

    # Then fixed RAG omits workflow decisions without losing answer scores
    assert_observation(fixed, "answer_token_f1", value=1.0, status="measured", sample_count=1)
    assert_observation(
        fixed,
        "route_accuracy",
        value=None,
        status="not_applicable",
        sample_count=0,
    )
    assert_observation(full, "route_accuracy", value=1.0, status="measured", sample_count=1)


def test_missing_full_rag_decisions_are_included_as_incorrect() -> None:
    metrics = aggregate_metrics(
        [case(id="complete"), case(id="missing")],
        [
            result(case_id="complete"),
            result(case_id="missing", route=None, strategy=None),
        ],
        system="full-rag",
    )

    assert_observation(metrics, "route_accuracy", value=0.5, status="measured", sample_count=2)
    assert_observation(metrics, "strategy_accuracy", value=0.5, status="measured", sample_count=2)


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
    assert_observation(metrics, "retry_precision", value=0.5, status="measured", sample_count=2)
    assert_observation(metrics, "retry_recall", value=1.0, status="measured", sample_count=1)

    missed = aggregate_metrics([case(expected_retry=True)], [result()])
    assert_observation(
        missed, "retry_precision", value=None, status="no_eligible_cases", sample_count=0
    )
    assert_observation(missed, "retry_recall", value=0.0, status="measured", sample_count=1)


def test_citation_metrics() -> None:
    assert citation_precision(["a", "unknown"], ["a", "b"]) == 0.5
    assert citation_precision([], ["a"]) is None
    assert gold_citation_coverage(["a"], ["a", "b"]) == 0.5
    assert gold_citation_coverage([], []) is None
    assert token_f1("Sam Altman [C1]", "Sam Altman") == 1.0


def test_citation_precision_uses_relevant_evidence_without_weakening_validation() -> None:
    benchmark = case(relevant_chunk_ids=["a", "b"])
    metrics = aggregate_metrics(
        [benchmark],
        [result(retrieved_chunk_ids=["a", "x"], cited_chunk_ids=["a", "x"])],
    )

    assert_observation(metrics, "citation_precision", value=0.5, status="measured", sample_count=2)
    assert "invalid_citation" in failure_labels(
        benchmark,
        result(retrieved_chunk_ids=["a", "x"], cited_chunk_ids=["a", "unknown"]),
    )


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
    assert_observation(metrics, "abstention_accuracy", value=1.0, status="measured", sample_count=2)
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
    assert_observation(metrics, "conflict_recall", value=1.0, status="measured", sample_count=1)
    assert_observation(
        metrics,
        "conflict_false_positive_rate",
        value=0.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(metrics, "termination_rate", value=0.5, status="measured", sample_count=2)
    assert "conflict_accuracy" not in metrics


def test_split_filtering() -> None:
    cases = [case(id="dev"), case(id="held", split="test")]
    assert [item.id for item in filter_cases(cases, "test")] == ["held"]
    assert [item.id for item in filter_cases(cases, "development", {"dev"})] == ["dev"]


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


def test_required_models_depend_on_selected_systems() -> None:
    # Act
    bm25 = evaluation.required_models_for_systems(["bm25"])
    dense_hybrid = evaluation.required_models_for_systems(["dense", "hybrid"])
    agentic = evaluation.required_models_for_systems(["full-rag"])
    alternate = evaluation.required_models_for_systems(["full-rag"], "qwen3:4b")

    # Then dependency checks include only the resources each selected system needs.
    assert bm25 == ()
    assert dense_hybrid == (evaluation.normalize_model_name(evaluation.config.embedding_model),)
    assert set(agentic) == {
        evaluation.normalize_model_name(evaluation.config.embedding_model),
        evaluation.normalize_model_name(evaluation.config.llm_model),
    }
    assert alternate == (
        evaluation.normalize_model_name(evaluation.config.embedding_model),
        "qwen3:4b",
    )


def test_ollama_model_matching_uses_exact_normalized_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def response(models: list[str]):
        return io.BytesIO(json.dumps({"models": [{"name": name} for name in models]}).encode())

    monkeypatch.setattr(
        evaluation.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: response(["qwen3.5:latest"]),
    )
    with pytest.raises(RuntimeError, match="qwen3.5:9b"):
        evaluation._require_ollama(["qwen3.5:9b"])

    monkeypatch.setattr(
        evaluation.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: response(["nomic-embed-text:latest"]),
    )
    evaluation._require_ollama(["nomic-embed-text"])


def test_bm25_evaluation_does_not_construct_ollama_or_agentic_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(case().model_dump_json() + "\n", encoding="utf-8")

    class Manager:
        def setup(self):
            return object()

    monkeypatch.setattr(evaluation, "VectorDBManager", Manager)
    monkeypatch.setattr(evaluation, "Retriever", lambda _collection: object())
    monkeypatch.setattr(
        evaluation,
        "_require_ollama",
        lambda *_args, **_kwargs: pytest.fail("BM25 must not require Ollama"),
    )
    monkeypatch.setattr(
        evaluation,
        "ChatOllama",
        lambda **_kwargs: pytest.fail("BM25 must not construct the chat model"),
    )
    monkeypatch.setattr(
        evaluation,
        "RAGGraph",
        lambda *_args, **_kwargs: pytest.fail("BM25 must not construct the agentic graph"),
    )
    monkeypatch.setattr(
        evaluation,
        "run_retrieval_case",
        lambda item, system, _retriever: result(case_id=item.id, system=system),
    )
    monkeypatch.setattr(evaluation, "write_experiment", lambda *_args, **_kwargs: tmp_path)

    assert evaluation.run_evaluation(dataset, ["bm25"], "development") == tmp_path


def test_run_evaluation_uses_normalized_run_scoped_chat_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(case().model_dump_json() + "\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class Manager:
        def setup(self):
            return object()

    class ChatModel:
        def __init__(self, **kwargs: object) -> None:
            captured["chat_kwargs"] = kwargs

    monkeypatch.setattr(evaluation, "VectorDBManager", Manager)
    monkeypatch.setattr(evaluation, "Retriever", lambda _collection: object())
    monkeypatch.setattr(evaluation, "ChatOllama", ChatModel)
    monkeypatch.setattr(
        evaluation,
        "RAGGraph",
        lambda manager, *, llm: {"manager": manager, "llm": llm},
    )
    monkeypatch.setattr(
        evaluation,
        "_require_ollama",
        lambda models: captured.update(required_models=models),
    )
    monkeypatch.setattr(
        evaluation,
        "run_agentic_case",
        lambda item, _graph, _model, *, timeout_seconds: result(
            case_id=item.id, system="full-rag", latency_seconds=timeout_seconds
        ),
    )

    def fake_write(
        _root: Path,
        experiment: ExperimentConfig,
        _results: list[CaseResult],
        _metrics: object,
    ) -> Path:
        captured["experiment"] = experiment
        return tmp_path / "result"

    monkeypatch.setattr(evaluation, "write_experiment", fake_write)

    # Act
    output = evaluation.run_evaluation(
        dataset,
        ["full-rag"],
        "development",
        chat_model=" qwen3 ",
    )

    # Then the normalized tag crosses preflight, construction, and metadata unchanged.
    assert output == tmp_path / "result"
    assert captured["required_models"] == (
        evaluation.normalize_model_name(evaluation.config.embedding_model),
        "qwen3:latest",
    )
    assert captured["chat_kwargs"] == {
        "model": "qwen3:latest",
        "base_url": evaluation.config.ollama_base_url,
        "temperature": evaluation.config.temperature,
        "num_predict": 512,
        "client_kwargs": {"timeout": evaluation.CANONICAL_REQUEST_TIMEOUT_SECONDS},
    }
    experiment = cast(ExperimentConfig, captured["experiment"])
    assert experiment.chat_model == "qwen3:latest"
    assert experiment.case_timeout_seconds == 30.0


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
        systems=["dense", "full-rag"],
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
            "full-rag": {
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
    assert set(summary) == {
        "benchmark_name",
        "case_ids",
        "completed_result_count",
        "configuration",
        "expected_result_count",
        "metrics",
        "result_kind",
    }
    assert summary["benchmark_name"] == "full_rag_benchmark"
    assert summary["case_ids"] == ["case-1"]
    assert summary["expected_result_count"] == 140
    assert summary["completed_result_count"] == 1
    assert summary["configuration"]["systems"] == ["dense", "full-rag"]
    assert summary["configuration"]["case_timeout_seconds"] == 30.0
    assert summary["metrics"]["full-rag"]["recall_at_5"] == {
        "value": 1.0,
        "status": "measured",
        "sample_count": 1,
        "note": "Cases with expected chunk evidence.",
    }
    assert (output / "cases.jsonl").read_text().strip()
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "full-rag" in markdown
    assert "1.000000 · n=1" in markdown
    assert "Cases with expected chunk evidence." in markdown


def canonical_experiment(**updates: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "run_id": "standard",
        "timestamp": "2026-01-02T03:04:05Z",
        "git_commit": "abc123",
        "dataset_hash": "sha256",
        "evaluated_split": "development",
        "systems": list(EXPECTED_SYSTEMS),
        "chat_model": "qwen3.5:9b",
        "embedding_model": "nomic-embed-text",
        "temperature": 0.0,
        "fixed_rag_prompt_id": "fixed_rag_grounded_answer",
        "chunk_size": 700,
        "chunk_overlap": 100,
        "retrieval_limit": 5,
        "semantic_candidates": 10,
        "sparse_candidates": 10,
        "retry_limit": 1,
        "subquery_limit": 4,
        "case_timeout_seconds": 30.0,
        "dataset_name": "multihop",
    }
    values.update(updates)
    return ExperimentConfig.model_validate(values)


def canonical_results() -> list[CaseResult]:
    return [
        result(case_id=case_id, system=system)
        for case_id in evaluation_models.CANONICAL_BENCHMARK_CASE_IDS
        for system in EXPECTED_SYSTEMS
    ]


def completeness_summary(
    experiment: ExperimentConfig,
    results: list[CaseResult],
) -> dict[str, object]:
    return {
        "benchmark_name": "full_rag_benchmark",
        "configuration": experiment.model_dump(mode="json"),
        "case_ids": list(dict.fromkeys(item.case_id for item in results)),
        "expected_result_count": 140,
        "completed_result_count": len(results),
    }


def test_complete_reordered_full_rag_benchmark_artifact_is_accepted() -> None:
    results = list(reversed(canonical_results()))
    summary = completeness_summary(canonical_experiment(), results)

    assert evaluation.is_complete_full_rag_benchmark_artifact(summary, results)
    assert evaluation.evaluation_result_kind(summary, results) == "standard_benchmark"


@pytest.mark.parametrize(
    "incomplete_results",
    [
        lambda items: [
            item
            for item in items
            if item.case_id != evaluation_models.CANONICAL_BENCHMARK_CASE_IDS[-1]
        ],
        lambda items: [item for item in items if item.system != EXPECTED_SYSTEMS[-1]],
        lambda items: [*items[:-1], items[0]],
        lambda items: [
            item
            for item in items
            if item.case_id == evaluation_models.CANONICAL_BENCHMARK_CASE_IDS[0]
        ],
    ],
    ids=["missing-case", "missing-system", "duplicate-pair", "one-case-seven-systems"],
)
def test_incomplete_full_rag_benchmark_artifact_is_custom(
    incomplete_results,
) -> None:
    results = incomplete_results(canonical_results())
    summary = completeness_summary(canonical_experiment(), results)

    assert not evaluation.is_complete_full_rag_benchmark_artifact(summary, results)
    assert evaluation.evaluation_result_kind(summary, results) == "custom_evaluation"


def test_incorrect_benchmark_configuration_is_custom() -> None:
    results = canonical_results()
    summary = completeness_summary(
        canonical_experiment(case_timeout_seconds=31.0),
        results,
    )

    assert not evaluation.is_complete_full_rag_benchmark_artifact(summary, results)


def test_exactly_140_canonical_results_are_accepted() -> None:
    results = canonical_results()
    summary = completeness_summary(canonical_experiment(), results)

    assert len(results) == 140
    assert evaluation.is_complete_full_rag_benchmark_artifact(summary, results)


def test_experiment_summary_records_standard_or_custom_kind(tmp_path: Path) -> None:
    base = {
        "timestamp": "2026-01-02T03:04:05Z",
        "git_commit": "abc123",
        "dataset_hash": "sha256",
        "chat_model": "chat",
        "embedding_model": "embed",
        "chunk_size": 700,
        "chunk_overlap": 100,
        "retrieval_limit": 5,
        "semantic_candidates": 10,
        "sparse_candidates": 10,
        "retry_limit": 1,
        "subquery_limit": 4,
        "dataset_name": "multihop",
    }
    metric = {
        "bm25": {"recall_at_5": MetricObservation(value=1.0, status="measured", sample_count=1)}
    }
    standard = canonical_experiment()
    custom = ExperimentConfig(
        run_id="custom",
        evaluated_split="development",
        systems=["bm25"],
        **base,
    )

    standard_path = write_experiment(tmp_path, standard, canonical_results(), metric)
    custom_path = write_experiment(tmp_path, custom, [result()], metric)

    assert json.loads((standard_path / "summary.json").read_text())["result_kind"] == (
        "standard_benchmark"
    )
    assert json.loads((custom_path / "summary.json").read_text())["result_kind"] == (
        "custom_evaluation"
    )


def test_cli_defaults_to_multihop_benchmark(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, list[str], str, str, str | None, float]] = []

    def fake_run(
        dataset: Path,
        systems: list[str],
        split: str,
        *,
        dataset_name: str,
        chat_model: str | None = None,
        case_timeout_seconds: float,
    ) -> Path:
        calls.append((dataset, systems, split, dataset_name, chat_model, case_timeout_seconds))
        return tmp_path / "result"

    monkeypatch.setattr(evaluation, "run_evaluation", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluation"])

    evaluation.main()

    assert calls == [
        (
            evaluation.MULTIHOP_ROOT / "cases.jsonl",
            list(evaluation.SYSTEMS),
            "development",
            "multihop",
            evaluation.config.llm_model,
            30.0,
        )
    ]

    monkeypatch.setattr(sys, "argv", ["evaluation", "--model", "qwen3:4b"])
    evaluation.main()

    assert calls[-1][-2] == "qwen3:4b"


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


def test_retrieval_baselines_do_not_receive_agent_only_failure_labels() -> None:
    labels = failure_labels(
        case(expected_retry=True, expected_conflict=True),
        result(
            system="dense",
            retrieved_chunk_ids=["x"],
            cited_chunk_ids=[],
            retry_count=0,
            conflict_detected=False,
        ),
    )

    assert labels == ["retrieval_miss"]


def test_fixed_rag_uses_exactly_one_grounded_model_call_and_validates_citations() -> None:
    # Arrange
    class FakeRetriever:
        def semantic(self, _question: str, _limit: int) -> list[evaluation.RetrievalHit]:
            return [
                evaluation.RetrievalHit(
                    chunk_id="a",
                    document_id="doc",
                    content="The answer is expected.",
                    filename="doc.pdf",
                    page=1,
                    score=0.9,
                )
            ]

    class FakeModel:
        calls = 0

        def invoke(self, _messages: object, **_kwargs: object) -> str:
            self.calls += 1
            return "Expected [C1]"

    model = FakeModel()
    # Act
    measured = run_fixed_rag_case(
        case(expected_answer="expected"),
        "dense-rag",
        cast(evaluation.Retriever, FakeRetriever()),
        cast(CountingModel, model),
    )

    # Then the answer remains grounded without a repair call
    assert model.calls == 1
    assert measured.answer == "Expected [C1]"
    assert measured.cited_chunk_ids == ["a"]
    assert measured.retrieved_chunk_ids == ["a"]
    assert measured.terminated is True
    assert measured.abstained is False
    assert measured.route is None
    assert measured.strategy is None


def test_fixed_rag_abstains_when_the_single_answer_cannot_be_grounded() -> None:
    # Arrange
    class FakeRetriever:
        def sparse(self, _question: str, _limit: int) -> list[evaluation.RetrievalHit]:
            return [
                evaluation.RetrievalHit(
                    chunk_id="a",
                    document_id="doc",
                    content="Grounded evidence.",
                    filename="doc.pdf",
                    page=1,
                    score=0.9,
                )
            ]

    class FakeModel:
        calls = 0

        def invoke(self, _messages: object, **_kwargs: object) -> str:
            self.calls += 1
            return "An unsupported answer without a citation."

    # Act
    measured = run_fixed_rag_case(
        case(),
        "bm25-rag",
        cast(evaluation.Retriever, FakeRetriever()),
        cast(CountingModel, FakeModel()),
    )

    # Then unsupported prose becomes an explicit abstention
    assert measured.abstained is True
    assert measured.cited_chunk_ids == []
    assert measured.validation_violations == ["uncited_claim"]
    assert "over_abstention" in measured.failure_labels


def test_agentic_case_records_decision_diagnostics() -> None:
    class DiagnosticGraph:
        def process_query(self, question: str, session_id: str) -> dict[str, object]:
            return {
                "answer": "Answer [C1]",
                "route": "complex_search",
                "strategy": "hybrid",
                "retry_count": 1,
                "conflict": True,
                "evidence_status": "limited",
                "subquery_specs": [
                    {"id": "SQ1", "text": "original"},
                    {"id": "SQ2", "text": "second"},
                ],
                "rewritten_subqueries": [{"id": "SQ1", "text": "refined"}],
                "supported_subquery_ids": ["SQ1"],
                "relevant_labels": ["C1"],
                "validation": {
                    "violations": [],
                    "initial_violations": ["uncited_claim"],
                    "repair_violations": [],
                },
                "retrieval_hits": [{"chunk_id": "a", "document_id": "doc"}],
                "sources": [{"chunk_id": "a"}],
                "trace": [
                    {"stage": "retrieve"},
                    {"stage": "retrieve"},
                    {"stage": "terminate", "termination": "limited"},
                ],
            }

    class FakeModel:
        calls = 4

    measured = run_agentic_case(
        case(expected_retry=True, expected_conflict=True),
        cast(evaluation.RAGGraph, DiagnosticGraph()),
        cast(CountingModel, FakeModel()),
    )

    assert measured.conflict_detected is True
    assert measured.evidence_status == "limited"
    assert measured.subquery_specs[0].id == "SQ1"
    assert measured.rewritten_subqueries[0].text == "refined"
    assert measured.supported_subquery_ids == ["SQ1"]
    assert measured.relevant_labels == ["C1"]
    assert measured.termination_reason == "limited"
    assert measured.initial_validation_violations == ["uncited_claim"]


def test_runtime_and_failed_abstention_labels() -> None:
    assert failure_labels(
        case(answerable=False, relevant_chunk_ids=[]),
        result(abstained=False, runtime_error="boom"),
    ) == [
        "failed_abstention",
        "runtime_error",
    ]


def test_runtime_errors_are_failed_responses_for_answer_metrics() -> None:
    metrics = aggregate_metrics(
        [case(expected_answer="expected answer")],
        [
            result(
                answer="expected answer",
                abstained=False,
                runtime_error="RuntimeError",
            )
        ],
    )

    assert_observation(metrics, "runtime_error_count", value=1.0, status="measured", sample_count=1)
    assert_observation(metrics, "runtime_error_rate", value=1.0, status="measured", sample_count=1)
    assert_observation(metrics, "abstention_accuracy", value=0.0, status="measured", sample_count=1)
    assert_observation(
        metrics, "answerable_response_rate", value=0.0, status="measured", sample_count=1
    )
    assert_observation(
        metrics,
        "normalized_answer_exact_match",
        value=0.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(metrics, "answer_token_f1", value=0.0, status="measured", sample_count=1)


def test_failure_classifications_are_aggregated_once_per_case() -> None:
    metrics = aggregate_metrics(
        [
            case(id="many", expected_retry=True, expected_conflict=True),
            case(id="failed-abstention", answerable=False, relevant_chunk_ids=[]),
        ],
        [
            result(
                case_id="many",
                route=None,
                strategy=None,
                retrieved_chunk_ids=["x"],
                cited_chunk_ids=["unknown"],
                retry_count=0,
                conflict_detected=False,
                terminated=False,
                abstained=True,
                runtime_error="RuntimeError",
            ),
            result(case_id="failed-abstention", abstained=False),
        ],
        system="full-rag",
    )

    for name in (
        "runtime_error_count",
        "retrieval_miss_count",
        "citation_failure_count",
        "over_abstention_count",
        "failed_abstention_count",
        "non_termination_count",
        "route_failure_count",
        "strategy_failure_count",
        "retry_failure_count",
        "conflict_failure_count",
    ):
        assert_observation(metrics, name, value=1.0, status="measured", sample_count=2)
    assert_observation(metrics, "runtime_error_rate", value=0.5, status="measured", sample_count=2)


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


def test_agentic_case_waits_for_an_active_model_request_to_exit() -> None:
    class SlowGraph:
        def process_query(self, question: str, session_id: str) -> dict[str, object]:
            time.sleep(0.1)
            return {}

    class FakeModel:
        calls = 0

    measured = run_agentic_case(
        case(),
        cast(evaluation.RAGGraph, SlowGraph()),
        cast(CountingModel, FakeModel()),
        timeout_seconds=0.01,
    )

    assert measured.runtime_error is None
    assert measured.latency_seconds >= 0.08


def test_counting_model_checks_cancellation_around_each_model_invocation() -> None:
    cancelled = False
    calls = 0

    class FakeModel:
        def invoke(self, value: object, **kwargs: object) -> str:
            nonlocal cancelled, calls
            del value, kwargs
            calls += 1
            cancelled = True
            return "completed response"

    model = CountingModel(FakeModel(), cancellation_check=lambda: cancelled)

    with pytest.raises(evaluation.EvaluationCancelled):
        model.invoke("first")
    with pytest.raises(evaluation.EvaluationCancelled):
        model.invoke("second")

    assert calls == 1


def test_validation_failure_is_counted_as_an_abstention() -> None:
    measured = result(
        answer="I could not produce a fully cited answer from the available evidence.",
        termination_reason="validation_failed",
    )

    assert evaluation.is_abstention_termination(measured.termination_reason)


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
    assert_observation(metrics, "recall_at_5", value=0.5, status="measured", sample_count=1)
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
    assert_observation(metrics, "answer_token_f1", value=1.0, status="measured", sample_count=1)
    assert normalized_exact_match("The Answer!", "the answer") == 1.0
    assert token_f1("alpha beta", "alpha gamma") == 0.5


def test_retrieval_only_response_metrics_are_not_applicable() -> None:
    metrics = aggregate_metrics([case()], [result(system="dense", route=None, strategy=None)])

    assert_observation(metrics, "recall_at_5", value=0.5, status="measured", sample_count=1)
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
        assert_observation(metrics, name, value=None, status="not_applicable", sample_count=0)


def test_conditioned_metrics_distinguish_empty_denominators_from_measured_zero() -> None:
    metrics = aggregate_metrics(
        [case(expected_answer="expected", expected_conflict=False)],
        [result(cited_chunk_ids=["unknown"], answer="different")],
    )

    assert_observation(metrics, "citation_precision", value=0.0, status="measured", sample_count=1)
    assert_observation(
        metrics,
        "gold_evidence_citation_coverage",
        value=0.0,
        status="measured",
        sample_count=1,
    )
    assert_observation(metrics, "answer_token_f1", value=0.0, status="measured", sample_count=1)
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
