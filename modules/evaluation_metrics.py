"""Pure metrics and failure classification for evaluation runs."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence

from modules.evaluation_models import (
    FAILURE_ORDER,
    CaseResult,
    EvaluationCase,
    MetricObservation,
    Split,
    SystemName,
)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], 1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 1.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], 1)
        if chunk_id in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])


def citation_precision(cited: Sequence[str], retrieved: Sequence[str]) -> float | None:
    if not cited:
        return None
    retrieved_ids = set(retrieved)
    return sum(chunk_id in retrieved_ids for chunk_id in cited) / len(cited)


def gold_citation_coverage(cited: Sequence[str], relevant: Sequence[str]) -> float | None:
    """Fraction of expected relevant chunk IDs cited (not claim-level coverage)."""
    gold = set(relevant)
    if not gold:
        return None
    return len(set(cited) & gold) / len(gold)


def _normalize_answer(value: str) -> str:
    value = re.sub(r"\[C\d+\]", " ", value, flags=re.IGNORECASE)
    value = value.lower().translate(str.maketrans({char: " " for char in string.punctuation}))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def normalized_exact_match(prediction: str, expected: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(expected))


def token_f1(prediction: str, expected: str) -> float:
    predicted = _normalize_answer(prediction).split()
    gold = _normalize_answer(expected).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _measured(value: float, sample_count: int, note: str) -> MetricObservation:
    return MetricObservation(
        value=value,
        status="measured",
        sample_count=sample_count,
        note=note,
    )


def _no_eligible(note: str) -> MetricObservation:
    return MetricObservation(
        value=None,
        status="no_eligible_cases",
        sample_count=0,
        note=note,
    )


def _not_applicable(note: str) -> MetricObservation:
    return MetricObservation(
        value=None,
        status="not_applicable",
        sample_count=0,
        note=note,
    )


def _mean_observation(values: Sequence[float], note: str) -> MetricObservation:
    if not values:
        return _no_eligible(note)
    return _measured(sum(values) / len(values), len(values), note)


def _ratio_observation(numerator: int, denominator: int, note: str) -> MetricObservation:
    if not denominator:
        return _no_eligible(note)
    return _measured(numerator / denominator, denominator, note)


def aggregate_metrics(
    cases: Sequence[EvaluationCase],
    results: Sequence[CaseResult],
    *,
    system: SystemName | None = None,
) -> dict[str, MetricObservation]:
    by_id = {item.id: item for item in cases}
    paired = [(by_id[item.case_id], item) for item in results if item.case_id in by_id]
    retrieval = [(case, result) for case, result in paired if case.relevant_chunk_ids]
    document_retrieval = [(case, result) for case, result in paired if case.relevant_document_ids]
    agentic = [(case, result) for case, result in paired if result.system == "agentic"]
    evaluated_system = system or (paired[0][1].system if paired else None)
    produces_answers = evaluated_system == "agentic"

    route = [(case, result) for case, result in agentic if result.route is not None]
    strategy = [(case, result) for case, result in agentic if result.strategy is not None]
    retry_tp = sum(case.expected_retry and result.retry_count > 0 for case, result in agentic)
    retry_fp = sum(not case.expected_retry and result.retry_count > 0 for case, result in agentic)
    retry_fn = sum(case.expected_retry and result.retry_count == 0 for case, result in agentic)
    retry_expected = retry_tp + retry_fn
    retry_predicted = retry_tp + retry_fp
    answer_pairs = [
        (case, result)
        for case, result in agentic
        if case.expected_answer is not None and case.answerable
    ]
    answerable = [(case, result) for case, result in agentic if case.answerable]
    unanswerable = [(case, result) for case, result in agentic if not case.answerable]
    conflict_positive = [(case, result) for case, result in agentic if case.expected_conflict]
    conflict_negative = [(case, result) for case, result in agentic if not case.expected_conflict]
    coverage_pairs = [
        (case, result) for case, result in agentic if case.answerable and case.relevant_chunk_ids
    ]
    emitted_citations = [
        (chunk_id, set(result.retrieved_chunk_ids))
        for _, result in agentic
        for chunk_id in result.cited_chunk_ids
    ]

    retrieval_note = "Cases with expected chunk evidence."
    response_only = "Retrieval-only systems do not produce answers or answer-level signals."
    metrics = {
        "recall_at_5": _mean_observation(
            [
                recall_at_k(result.retrieved_chunk_ids, set(case.relevant_chunk_ids))
                for case, result in retrieval
            ],
            retrieval_note,
        ),
        "document_recall_at_5": _mean_observation(
            [
                recall_at_k(result.retrieved_document_ids, set(case.relevant_document_ids))
                for case, result in document_retrieval
            ],
            "Cases with expected document evidence.",
        ),
        "mrr_at_5": _mean_observation(
            [
                mrr_at_k(result.retrieved_chunk_ids, set(case.relevant_chunk_ids))
                for case, result in retrieval
            ],
            retrieval_note,
        ),
        "ndcg_at_5": _mean_observation(
            [
                ndcg_at_k(result.retrieved_chunk_ids, set(case.relevant_chunk_ids))
                for case, result in retrieval
            ],
            retrieval_note,
        ),
        "termination_rate": _ratio_observation(
            sum(result.terminated for _, result in paired),
            len(paired),
            "All evaluated cases.",
        ),
        "mean_latency_seconds": _mean_observation(
            [result.latency_seconds for _, result in paired], "All evaluated cases."
        ),
        "p95_latency_seconds": (
            _measured(
                p95([result.latency_seconds for _, result in paired]),
                len(paired),
                "All evaluated cases.",
            )
            if paired
            else _no_eligible("All evaluated cases.")
        ),
        "mean_llm_calls_per_query": _mean_observation(
            [float(result.llm_calls) for _, result in paired], "All evaluated cases."
        ),
        "mean_retrieval_rounds_per_query": _mean_observation(
            [float(result.retrieval_rounds) for _, result in paired],
            "All evaluated cases.",
        ),
    }

    if not produces_answers:
        metrics.update(
            {
                name: _not_applicable(response_only)
                for name in (
                    "route_accuracy",
                    "strategy_accuracy",
                    "retry_precision",
                    "retry_recall",
                    "citation_precision",
                    "gold_evidence_citation_coverage",
                    "abstention_accuracy",
                    "unanswerable_abstention_recall",
                    "answerable_response_rate",
                    "conflict_recall",
                    "conflict_false_positive_rate",
                    "normalized_answer_exact_match",
                    "answer_token_f1",
                )
            }
        )
        return metrics

    metrics.update(
        {
            "route_accuracy": _ratio_observation(
                sum(result.route == case.expected_route for case, result in route),
                len(route),
                "Agentic cases with a recorded route.",
            ),
            "strategy_accuracy": _ratio_observation(
                sum(result.strategy == case.expected_strategy for case, result in strategy),
                len(strategy),
                "Agentic cases with a recorded retrieval strategy.",
            ),
            "retry_precision": _ratio_observation(
                retry_tp, retry_predicted, "Agentic cases where a retry was attempted."
            ),
            "retry_recall": _ratio_observation(
                retry_tp, retry_expected, "Agentic cases where a retry was expected."
            ),
            "citation_precision": _ratio_observation(
                sum(chunk_id in retrieved for chunk_id, retrieved in emitted_citations),
                len(emitted_citations),
                "Citations emitted by agentic answers.",
            ),
            "gold_evidence_citation_coverage": _mean_observation(
                [
                    coverage
                    for case, result in coverage_pairs
                    if (
                        coverage := gold_citation_coverage(
                            result.cited_chunk_ids, case.relevant_chunk_ids
                        )
                    )
                    is not None
                ],
                "Answerable cases with expected chunk evidence.",
            ),
            "abstention_accuracy": _ratio_observation(
                sum(result.abstained == (not case.answerable) for case, result in agentic),
                len(agentic),
                "All agentic cases.",
            ),
            "unanswerable_abstention_recall": _ratio_observation(
                sum(result.abstained for _, result in unanswerable),
                len(unanswerable),
                "Unanswerable agentic cases.",
            ),
            "answerable_response_rate": _ratio_observation(
                sum(not result.abstained for _, result in answerable),
                len(answerable),
                "Answerable agentic cases.",
            ),
            "conflict_recall": _ratio_observation(
                sum(result.conflict_detected for _, result in conflict_positive),
                len(conflict_positive),
                "Agentic cases expected to contain a conflict.",
            ),
            "conflict_false_positive_rate": _ratio_observation(
                sum(result.conflict_detected for _, result in conflict_negative),
                len(conflict_negative),
                "Agentic cases not expected to contain a conflict.",
            ),
            "normalized_answer_exact_match": _mean_observation(
                [
                    normalized_exact_match(result.answer, case.expected_answer or "")
                    for case, result in answer_pairs
                ],
                "Answerable agentic cases with an expected answer.",
            ),
            "answer_token_f1": _mean_observation(
                [
                    token_f1(result.answer, case.expected_answer or "")
                    for case, result in answer_pairs
                ],
                "Answerable agentic cases with an expected answer.",
            ),
        }
    )
    return metrics


def failure_labels(case: EvaluationCase, result: CaseResult) -> list[str]:
    labels: set[str] = set()
    if case.relevant_chunk_ids and not (
        set(result.retrieved_chunk_ids) & set(case.relevant_chunk_ids)
    ):
        labels.add("retrieval_miss")
    if result.system == "agentic":
        if result.route is not None and result.route != case.expected_route:
            labels.add("route_error")
        if result.strategy is not None and result.strategy != case.expected_strategy:
            labels.add("strategy_error")
        if set(result.cited_chunk_ids) - set(result.retrieved_chunk_ids):
            labels.add("invalid_citation")
        coverage = gold_citation_coverage(result.cited_chunk_ids, case.relevant_chunk_ids)
        if coverage is not None and coverage < 1:
            labels.add("citation_coverage_miss")
        if case.answerable and result.abstained:
            labels.add("over_abstention")
        if not case.answerable and not result.abstained:
            labels.add("failed_abstention")
        if (result.retry_count > 0) != case.expected_retry:
            labels.add("retry_error")
        if result.conflict_detected != case.expected_conflict:
            labels.add("conflict_miss")
        if not result.terminated:
            labels.add("non_termination")
        if result.runtime_error:
            labels.add("runtime_error")
    return [label for label in FAILURE_ORDER if label in labels]


def filter_cases(
    cases: Sequence[EvaluationCase],
    split: Split,
    case_ids: set[str] | None = None,
) -> list[EvaluationCase]:
    return [
        case for case in cases if case.split == split and (case_ids is None or case.id in case_ids)
    ]
