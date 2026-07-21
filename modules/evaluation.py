"""Small comparative evaluation harness for the four MVP RAG systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, model_validator

from modules.config import PROJECT_ROOT, Settings, config
from modules.models import RetrievalHit
from modules.rag_graph import RAGGraph
from modules.retrieval import Retriever, reciprocal_rank_fusion
from modules.vector_db import VectorDBManager

Split = Literal["development", "test"]
SystemName = Literal["dense", "bm25", "hybrid", "agentic"]
MetricStatus = Literal["measured", "not_applicable", "no_eligible_cases"]
EvaluationResultKind = Literal["standard_benchmark", "custom_evaluation"]
SYSTEMS: tuple[SystemName, ...] = ("dense", "bm25", "hybrid", "agentic")
STANDARD_BENCHMARK_DATASET = "multihop"
STANDARD_BENCHMARK_SPLIT: Split = "development"
FAILURE_ORDER = (
    "route_error",
    "strategy_error",
    "retrieval_miss",
    "invalid_citation",
    "citation_coverage_miss",
    "over_abstention",
    "failed_abstention",
    "retry_error",
    "conflict_miss",
    "non_termination",
    "runtime_error",
)

MULTIHOP_ROOT = PROJECT_ROOT / "evals" / "multihop"


def is_standard_benchmark_summary(summary: dict[str, Any]) -> bool:
    """Return whether a schema-v2 result satisfies the canonical benchmark contract."""
    if summary.get("schema_version") != 2:
        return False
    configuration = summary.get("configuration")
    if not isinstance(configuration, dict):
        return False
    systems = configuration.get("systems")
    if not isinstance(systems, list):
        return False
    return (
        configuration.get("dataset_name") == STANDARD_BENCHMARK_DATASET
        and configuration.get("evaluated_split") == STANDARD_BENCHMARK_SPLIT
        and len(systems) == len(SYSTEMS)
        and set(systems) == set(SYSTEMS)
    )


def evaluation_result_kind(summary: dict[str, Any]) -> EvaluationResultKind:
    """Classify compatible results without presenting partial runs as benchmarks."""
    return "standard_benchmark" if is_standard_benchmark_summary(summary) else "custom_evaluation"


class BenchmarkEvidence(BaseModel):
    benchmark_document_id: str
    source: str = ""
    title: str = ""
    url: str = ""
    evidence_text: str
    author: str = ""
    category: str = ""
    published_at: str = ""


class EvaluationCase(BaseModel):
    id: str
    split: Split
    category: str
    question: str
    answerable: bool
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[BenchmarkEvidence] = Field(default_factory=list)
    expected_answer: str | None = None
    expected_route: str
    expected_strategy: str
    expected_retry: bool
    expected_conflict: bool


class CaseResult(BaseModel):
    case_id: str
    system: SystemName
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    route: str | None = None
    strategy: str | None = None
    retry_count: int = Field(default=0, ge=0)
    conflict_detected: bool = False
    terminated: bool = False
    abstained: bool = False
    latency_seconds: float = Field(default=0.0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    retrieval_rounds: int = Field(default=0, ge=0)
    failure_labels: list[str] = Field(default_factory=list)
    runtime_error: str | None = None
    answer: str = ""


class ExperimentConfig(BaseModel):
    run_id: str
    timestamp: str
    git_commit: str
    dataset_hash: str
    evaluated_split: Split
    systems: list[SystemName]
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_limit: int
    semantic_candidates: int
    sparse_candidates: int
    retry_limit: int
    subquery_limit: int
    dataset_name: str = "custom"
    dataset_version: str = "local"
    dataset_license: str = "unspecified"


class MetricObservation(BaseModel):
    """A metric value together with its applicability and evaluated denominator."""

    value: float | None
    status: MetricStatus
    sample_count: int = Field(ge=0)
    note: str | None = None

    @model_validator(mode="after")
    def validate_value_matches_status(self) -> Self:
        if self.status == "measured" and self.value is None:
            raise ValueError("measured metrics require a value")
        if self.status != "measured" and self.value is not None:
            raise ValueError("unmeasured metrics must use a null value")
        return self


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
    if result.route is not None and result.route != case.expected_route:
        labels.add("route_error")
    if result.strategy is not None and result.strategy != case.expected_strategy:
        labels.add("strategy_error")
    if case.relevant_chunk_ids and not (
        set(result.retrieved_chunk_ids) & set(case.relevant_chunk_ids)
    ):
        labels.add("retrieval_miss")
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


def filter_cases(cases: Sequence[EvaluationCase], split: Split) -> list[EvaluationCase]:
    return [case for case in cases if case.split == split]


class _CallState:
    calls = 0


class CountingModel:
    """Thin proxy counting all calls, including structured-output calls, at one boundary."""

    def __init__(self, model: Any, state: _CallState | None = None):
        self._model = model
        self._state = state or _CallState()

    @property
    def calls(self) -> int:
        return self._state.calls

    def invoke(self, value: object, **kwargs: object) -> Any:
        self._state.calls += 1
        return self._model.invoke(value, **kwargs)

    def with_structured_output(self, schema: object, **kwargs: object) -> CountingModel:
        return CountingModel(self._model.with_structured_output(schema, **kwargs), self._state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def load_cases(path: Path) -> list[EvaluationCase]:
    return [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _retrieval_result(
    case: EvaluationCase, system: SystemName, hits: list[RetrievalHit], elapsed: float
) -> CaseResult:
    result = CaseResult(
        case_id=case.id,
        system=system,
        retrieved_chunk_ids=[hit.chunk_id for hit in hits[:5]],
        retrieved_document_ids=list(
            dict.fromkeys(hit.document_id for hit in hits[:5] if hit.document_id)
        ),
        terminated=True,
        abstained=not bool(hits),
        latency_seconds=elapsed,
        retrieval_rounds=1,
    )
    result.failure_labels = failure_labels(case, result)
    return result


def run_retrieval_case(
    case: EvaluationCase, system: SystemName, retriever: Retriever
) -> CaseResult:
    started = time.perf_counter()
    if system == "dense":
        hits = retriever.semantic(case.question, 5)
    elif system == "bm25":
        hits = retriever.sparse(case.question, 5)
    elif system == "hybrid":
        dense = retriever.semantic(case.question, config.semantic_candidates)
        sparse = retriever.sparse(case.question, config.sparse_candidates)
        hits = reciprocal_rank_fusion(dense, sparse, limit=5)
    else:
        raise ValueError(f"Unsupported retrieval-only system: {system}")
    return _retrieval_result(case, system, hits, time.perf_counter() - started)


def run_agentic_case(case: EvaluationCase, graph: RAGGraph, model: CountingModel) -> CaseResult:
    started_calls = model.calls
    started = time.perf_counter()
    try:
        payload = graph.process_query(case.question, f"evaluation-{case.id}")
    except Exception as exc:  # noqa: BLE001 - evaluation must preserve per-case failures
        result = CaseResult(
            case_id=case.id,
            system="agentic",
            latency_seconds=time.perf_counter() - started,
            llm_calls=model.calls - started_calls,
            runtime_error=f"{type(exc).__name__}: {exc}",
        )
        result.failure_labels = failure_labels(case, result)
        return result
    trace = payload.get("trace", [])
    termination = next(
        (
            event.get("termination")
            for event in reversed(trace)
            if event.get("stage") == "terminate"
        ),
        None,
    )
    result = CaseResult(
        case_id=case.id,
        system="agentic",
        retrieved_chunk_ids=[hit["chunk_id"] for hit in payload.get("retrieval_hits", [])][:5],
        retrieved_document_ids=list(
            dict.fromkeys(
                hit.get("document_id", "")
                for hit in payload.get("retrieval_hits", [])[:5]
                if hit.get("document_id")
            )
        ),
        cited_chunk_ids=[source["chunk_id"] for source in payload.get("sources", [])],
        route=payload.get("route"),
        strategy=payload.get("strategy"),
        retry_count=payload.get("retry_count", 0),
        conflict_detected=False,
        terminated=termination is not None,
        abstained=termination in {"unsupported", "out_of_scope", "clarification"},
        latency_seconds=time.perf_counter() - started,
        llm_calls=model.calls - started_calls,
        retrieval_rounds=sum(event.get("stage") == "retrieve" for event in trace),
        answer=str(payload.get("answer", "")),
    )
    result.failure_labels = failure_labels(case, result)
    return result


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def write_experiment(
    results_root: Path,
    experiment: ExperimentConfig,
    results: Sequence[CaseResult],
    metrics: dict[str, dict[str, MetricObservation]],
) -> Path:
    output = results_root / experiment.run_id
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")
    serialized_metrics = {
        system: {name: observation.model_dump(mode="json") for name, observation in values.items()}
        for system, values in metrics.items()
    }
    configuration = experiment.model_dump(mode="json")
    summary_core = {
        "schema_version": 2,
        "configuration": configuration,
    }
    summary = {
        **summary_core,
        "result_kind": evaluation_result_kind(summary_core),
        "metrics": serialized_metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    result_label = (
        "Standard benchmark"
        if summary["result_kind"] == "standard_benchmark"
        else "Custom evaluation"
    )
    lines = [
        f"# Evaluation {experiment.run_id}",
        "",
        f"Result: **{result_label}**",
        "",
        f"Split: `{experiment.evaluated_split}`",
        "",
    ]
    for system, values in metrics.items():
        lines.extend([f"## {system}", ""])
        for name, observation in values.items():
            if observation.status == "measured":
                display = f"{observation.value:.6f} · n={observation.sample_count}"
            elif observation.status == "not_applicable":
                display = "Not applicable"
            else:
                display = "No eligible cases"
            note = f" — {observation.note}" if observation.note else ""
            lines.append(f"- {name}: {display}{note}")
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def normalize_model_name(name: str) -> str:
    """Return the exact Ollama identifier, adding the implicit latest tag."""
    normalized = name.strip()
    return normalized if ":" in normalized else f"{normalized}:latest"


def required_models_for_systems(systems: Sequence[SystemName]) -> tuple[str, ...]:
    """Return only the local models needed by the selected evaluation systems."""
    required: list[str] = []
    if any(system in {"dense", "hybrid", "agentic"} for system in systems):
        required.append(normalize_model_name(config.embedding_model))
    if "agentic" in systems:
        required.append(normalize_model_name(config.llm_model))
    return tuple(dict.fromkeys(required))


def _require_ollama(required_models: Sequence[str] | None = None) -> None:
    try:
        with urllib.request.urlopen(f"{config.ollama_base_url}/api/tags", timeout=3) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Live evaluation requires Ollama at {config.ollama_base_url}. "
            "Start Ollama and pull the configured chat and embedding models."
        ) from exc
    available = {
        normalize_model_name(str(item.get("name", "")))
        for item in payload.get("models", [])
        if item.get("name")
    }
    configured = required_models or (config.llm_model, config.embedding_model)
    required = {normalize_model_name(name) for name in configured}
    missing = sorted(required - available)
    if missing:
        commands = ", ".join(f"ollama pull {name}" for name in missing)
        raise RuntimeError(
            f"Missing required Ollama model(s): {', '.join(missing)}. Run: {commands}"
        )


def multihop_settings() -> Settings:
    runtime = PROJECT_ROOT / "evals" / "runtime"
    return Settings(
        sources_dir=MULTIHOP_ROOT / "corpus",
        data_dir=runtime,
        chroma_dir=runtime / "chroma",
        manifest_path=runtime / "manifest.json",
        trace_dir=runtime / "traces",
        logs_dir=runtime / "logs",
    )


def _evidence_matches(evidence: str, chunk: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    return normalize(evidence) in normalize(chunk)


def preflight_multihop(
    cases: Sequence[EvaluationCase],
    source_map_path: Path = MULTIHOP_ROOT / "source_map.json",
    manager: VectorDBManager | None = None,
    *,
    check_models: bool = True,
) -> list[EvaluationCase]:
    """Resolve stable benchmark evidence before any live-model invocation."""
    cases_path = source_map_path.with_name("cases.jsonl")
    if not source_map_path.exists() or not cases_path.exists():
        raise RuntimeError(
            "MultiHop-RAG benchmark files are missing. Run: "
            "uv run python scripts/prepare_multihop_eval.py --index"
        )
    payload = json.loads(source_map_path.read_text(encoding="utf-8"))
    source_map = payload.get("documents", payload)
    active_manager = manager or VectorDBManager(multihop_settings())
    manifest = active_manager.manifest()
    if not manifest.documents or active_manager.chunk_count() == 0:
        raise RuntimeError(
            "MultiHop-RAG manifest and Chroma must be populated. Run the preparation script "
            "with --index."
        )
    dev_ids = {case.id for case in cases if case.split == "development"}
    test_ids = {case.id for case in cases if case.split == "test"}
    if dev_ids & test_ids:
        raise RuntimeError("MultiHop-RAG development and held-out IDs overlap.")
    collection = active_manager.setup().get(include=["documents", "metadatas"])
    chunk_rows = list(
        zip(
            collection.get("ids", []),
            collection.get("documents", []) or [],
            collection.get("metadatas", []) or [],
            strict=True,
        )
    )
    resolved: list[EvaluationCase] = []
    failures: list[str] = []
    for case in cases:
        chunk_ids: list[str] = []
        document_ids: list[str] = []
        for evidence in case.gold_evidence:
            record = source_map.get(evidence.benchmark_document_id)
            if not record:
                failures.append(f"{case.id}: unknown document {evidence.benchmark_document_id}")
                continue
            document_id = str(record["document_id"])
            matches = [
                str(chunk_id)
                for chunk_id, text, metadata in chunk_rows
                if str(metadata.get("document_id", "")) == document_id
                and _evidence_matches(evidence.evidence_text, str(text))
            ]
            if not matches:
                failures.append(
                    f"{case.id}: evidence not found in indexed document "
                    f"{evidence.benchmark_document_id}"
                )
                continue
            document_ids.append(document_id)
            chunk_ids.extend(matches)
        resolved.append(
            case.model_copy(
                update={
                    "relevant_chunk_ids": list(dict.fromkeys(chunk_ids)),
                    "relevant_document_ids": list(dict.fromkeys(document_ids)),
                }
            )
        )
    if failures:
        raise RuntimeError("Gold evidence resolution failed:\n- " + "\n- ".join(failures))
    if check_models:
        _require_ollama()
    return resolved


def run_evaluation(
    dataset: Path, systems: Sequence[SystemName], split: Split, *, dataset_name: str = "custom"
) -> Path:
    selected = tuple(dict.fromkeys(systems))
    if not selected:
        raise ValueError("Select at least one evaluation system.")
    raw = dataset.read_bytes()
    all_cases = load_cases(dataset)
    if dataset_name == "multihop":
        manager = VectorDBManager(multihop_settings())
        all_cases = preflight_multihop(all_cases, manager=manager, check_models=False)
    else:
        manager = VectorDBManager()
    required_models = required_models_for_systems(selected)
    if required_models:
        _require_ollama(required_models)
    cases = filter_cases(all_cases, split)
    if not cases:
        raise ValueError(f"Dataset has no cases for split '{split}'")
    retriever = Retriever(manager.setup())
    counted: CountingModel | None = None
    graph: RAGGraph | None = None
    if "agentic" in selected:
        counted = CountingModel(
            ChatOllama(
                model=config.llm_model,
                base_url=config.ollama_base_url,
                temperature=config.temperature,
            )
        )
        graph = RAGGraph(manager, llm=counted)  # type: ignore[arg-type]
    results: list[CaseResult] = []
    for system in selected:
        for case in cases:
            results.append(
                run_agentic_case(case, graph, counted)  # type: ignore[arg-type]
                if system == "agentic"
                else run_retrieval_case(case, system, retriever)
            )
    metrics = {
        system: aggregate_metrics(
            cases,
            [item for item in results if item.system == system],
            system=system,
        )
        for system in selected
    }
    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-{split}"
    experiment = ExperimentConfig(
        run_id=run_id,
        timestamp=now.isoformat(),
        git_commit=_git_commit(),
        dataset_hash=hashlib.sha256(raw).hexdigest(),
        evaluated_split=split,
        systems=list(selected),
        chat_model=config.llm_model,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        retrieval_limit=5,
        semantic_candidates=config.semantic_candidates,
        sparse_candidates=config.sparse_candidates,
        retry_limit=config.max_retries,
        subquery_limit=config.max_subqueries,
        dataset_name=dataset_name,
        dataset_version=("yixuantt/MultiHopRAG" if dataset_name == "multihop" else "local"),
        dataset_license=("ODC-By-1.0" if dataset_name == "multihop" else "unspecified"),
    )
    return write_experiment(
        PROJECT_ROOT / "evals" / "results" / dataset_name, experiment, results, metrics
    )


def _parse_systems(value: str) -> list[SystemName]:
    requested = list(SYSTEMS) if value == "all" else [item.strip() for item in value.split(",")]
    invalid = sorted(set(requested) - set(SYSTEMS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown system(s): {', '.join(invalid)}")
    return list(dict.fromkeys(requested))  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=_parse_systems, default=list(SYSTEMS))
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--dataset", default="multihop")
    args = parser.parse_args()
    try:
        dataset_name = str(args.dataset)
        if dataset_name == "multihop":
            dataset = MULTIHOP_ROOT / "cases.jsonl"
        elif dataset_name in {"regression", "mvp"}:
            dataset = PROJECT_ROOT / "evals" / "mvp_cases.jsonl"
            dataset_name = "regression"
        else:
            dataset = Path(dataset_name)
            dataset_name = "custom"
        output = run_evaluation(dataset, args.systems, args.split, dataset_name=dataset_name)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"evaluation error: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
