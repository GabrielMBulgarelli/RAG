"""Shared formatting for evaluation metric observations."""

from __future__ import annotations

from modules.ui.contracts import EvaluationMetricObservation

_PERCENTAGE_METRICS = {
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "document_recall_at_5",
    "citation_precision",
    "gold_evidence_citation_coverage",
    "abstention_accuracy",
    "unanswerable_abstention_recall",
    "answerable_response_rate",
    "conflict_recall",
    "conflict_false_positive_rate",
    "normalized_answer_exact_match",
    "answer_token_f1",
}


def format_evaluation_observation(
    *,
    name: str,
    observation: EvaluationMetricObservation,
) -> str:
    """Preserve status, scale, and sample-count semantics in a compact value."""
    if observation.status == "not_applicable":
        return "Not applicable"
    if observation.status == "no_eligible_cases":
        return "No eligible cases"
    if observation.status != "measured" or observation.value is None:
        return "Missing observation"
    if name in _PERCENTAGE_METRICS:
        value = f"{observation.value * 100:.1f}%"
    elif name == "p95_latency_seconds":
        value = f"{round(observation.value * 1000):,} ms"
    else:
        value = f"{observation.value:.1f}"
    if observation.sample_count is not None:
        value += f" · n={observation.sample_count}"
    return value
