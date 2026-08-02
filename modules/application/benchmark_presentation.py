"""Transform benchmark evaluation results into application presentation models."""

from collections.abc import Mapping, Sequence
from typing import cast

from modules.application.models import (
    BenchmarkCaseDetail,
    BenchmarkCaseMetricObservation,
    BenchmarkMetric,
    BenchmarkMetricObservation,
    BenchmarkSection,
    TraceEvent,
)
from modules.evaluation import map_retrieved_evidence
from modules.evaluation_metrics import aggregate_metrics
from modules.evaluation_models import (
    ANSWER_SYSTEMS,
    SYSTEMS,
    CaseResult,
    EvaluationCase,
    MetricObservation,
    SystemName,
)

_METRIC_LABELS = {
    "recall_at_5": "Recall at 5",
    "document_recall_at_5": "Document recall at 5",
    "mrr_at_5": "MRR at 5",
    "ndcg_at_5": "nDCG at 5",
    "citation_precision": "Citation precision",
    "gold_evidence_citation_coverage": "Evidence coverage",
    "abstention_accuracy": "Abstention accuracy",
    "unanswerable_abstention_recall": "Unanswerable abstention recall",
    "answerable_response_rate": "Answerable response rate",
    "normalized_answer_exact_match": "Normalized exact match",
    "answer_token_f1": "Answer token F1",  # lanorme: ignore[SECRETPY-001] -- metric name
    "termination_rate": "Termination rate",
    "mean_latency_seconds": "Mean latency",
    "p95_latency_seconds": "P95 latency",
    "mean_llm_calls_per_query": "Mean LLM calls",
    "mean_retrieval_rounds_per_query": "Mean retrieval rounds",
    "route_accuracy": "Route accuracy",
    "strategy_accuracy": "Strategy accuracy",
    "retry_precision": "Retry precision",
    "retry_recall": "Retry recall",
    "conflict_recall": "Conflict recall",
    "conflict_false_positive_rate": "Conflict false-positive rate",
    "runtime_error_count": "Runtime error count",
    "runtime_error_rate": "Runtime error rate",
    "retrieval_miss_count": "Retrieval miss count",
    "citation_failure_count": "Citation failure count",
    "over_abstention_count": "Over-abstention count",
    "failed_abstention_count": "Failed abstention count",
    "non_termination_count": "Non-termination count",
    "route_failure_count": "Route failure count",
    "strategy_failure_count": "Strategy failure count",
    "retry_failure_count": "Retry failure count",
    "conflict_failure_count": "Conflict failure count",
}
_SECTION_METRICS = {
    "retrieval": ("recall_at_5", "document_recall_at_5", "mrr_at_5", "ndcg_at_5"),
    "grounding": (
        "citation_precision",
        "gold_evidence_citation_coverage",
        "abstention_accuracy",
        "unanswerable_abstention_recall",
        "answerable_response_rate",
        "normalized_answer_exact_match",
        "answer_token_f1",
    ),
    "execution": (
        "termination_rate",
        "mean_latency_seconds",
        "p95_latency_seconds",
        "mean_llm_calls_per_query",
        "mean_retrieval_rounds_per_query",
        "route_accuracy",
        "strategy_accuracy",
        "retry_precision",
        "retry_recall",
        "conflict_recall",
        "conflict_false_positive_rate",
        "runtime_error_count",
        "runtime_error_rate",
        "retrieval_miss_count",
        "citation_failure_count",
        "over_abstention_count",
        "failed_abstention_count",
        "non_termination_count",
        "route_failure_count",
        "strategy_failure_count",
        "retry_failure_count",
        "conflict_failure_count",
    ),
}


def _metric(
    *,
    name: str,
    observations: dict[SystemName, dict[str, MetricObservation]],
    systems: Sequence[SystemName],
) -> BenchmarkMetric:
    return BenchmarkMetric(
        name=name,
        label=_METRIC_LABELS.get(name, name.replace("_", " ").title()),
        observations=[
            BenchmarkMetricObservation(
                system=system,
                **observations[system][name].model_dump(),
            )
            for system in systems
        ],
    )


def benchmark_sections(
    observations: dict[SystemName, dict[str, MetricObservation]],
) -> list[BenchmarkSection]:
    specifications = (
        (
            "retrieval",
            "Retrieval",
            SYSTEMS,
            "Retrieval quality at five for every system's retrieval stage.",
        ),
        (
            "grounding",
            "Grounding",
            ANSWER_SYSTEMS,
            "Answer quality and evidence grounding for answer-producing systems.",
        ),
        (
            "execution",
            "Execution",
            SYSTEMS,
            "Runtime characteristics and full-RAG workflow decisions.",
        ),
    )
    return [
        BenchmarkSection(
            id=section_id,
            title=title,
            system_ids=list(systems),
            metrics=[
                _metric(name=name, observations=observations, systems=systems)
                for name in _SECTION_METRICS[section_id]
            ],
            detail=detail,
        )
        for section_id, title, systems, detail in specifications
    ]


def _public_trace_event(event: Mapping[str, object]) -> TraceEvent:
    return TraceEvent(
        stage=str(event.get("stage") or "evaluate"),
        decision=str(event.get("decision") or "completed"),
        retrieved_count=cast(int | None, event.get("retrieved_count")),
        fused_count=cast(int | None, event.get("fused_count")),
        selected_count=cast(int | None, event.get("selected_count")),
        retry_count=cast(int, event.get("retry_count") or 0),
        llm_calls=cast(int, event.get("llm_calls") or 0),
        termination=str(event.get("termination") or "continue"),
        duration_ms=cast(float | None, event.get("duration_ms")),
    )


def _trace(result: CaseResult) -> list[TraceEvent]:
    if result.public_trace:
        return [_public_trace_event(event) for event in result.public_trace]
    return [
        TraceEvent(
            stage="evaluate",
            decision="completed" if result.terminated else "failed",
            retrieved_count=len(result.retrieved_chunk_ids),
            fused_count=None,
            selected_count=len(result.retrieved_chunk_ids),
            retry_count=result.retry_count,
            llm_calls=result.llm_calls,
            termination=result.termination_reason
            or ("complete" if result.terminated else "failed"),
            duration_ms=result.latency_seconds * 1000,
        )
    ]


def benchmark_case_detail(*, case: EvaluationCase, result: CaseResult) -> BenchmarkCaseDetail:
    metrics = aggregate_metrics([case], [result], system=result.system)
    expected_evidence = (
        [item.model_dump(mode="json") for item in case.gold_evidence]
        if case.gold_evidence
        else [
            {"chunk_id": chunk_id, "document_ids": list(case.relevant_document_ids)}
            for chunk_id in case.relevant_chunk_ids
        ]
    )
    return BenchmarkCaseDetail(
        case_id=case.id,
        system=result.system,
        question=case.question,
        expected_answer=case.expected_answer,
        generated_answer=result.answer or None,
        expected_evidence=expected_evidence,
        retrieved_evidence=map_retrieved_evidence(
            result.retrieved_evidence
            or [{"chunk_id": chunk_id} for chunk_id in result.retrieved_chunk_ids]
        ),
        metric_observations=[
            BenchmarkCaseMetricObservation(
                name=name,
                label=_METRIC_LABELS.get(name, name.replace("_", " ").title()),
                system=result.system,
                **observation.model_dump(),
            )
            for name, observation in metrics.items()
        ],
        failure_classification=", ".join(result.failure_labels) or None,
        public_trace=_trace(result),
        sanitized_raw_result={
            "failure_labels": list(result.failure_labels),
            "retrieval_rounds": result.retrieval_rounds,
            "llm_calls": result.llm_calls,
            "terminated": result.terminated,
            "abstained": result.abstained,
        },
    )
