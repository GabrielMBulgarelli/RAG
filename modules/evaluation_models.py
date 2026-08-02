"""Evaluation schemas and benchmark contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, Field, JsonValue, model_validator

from modules.config import PROJECT_ROOT

Split = Literal["development", "test"]
SystemName = Literal[
    "dense",
    "bm25",
    "hybrid",
    "dense-rag",
    "bm25-rag",
    "hybrid-rag",
    "full-rag",
]
MetricStatus = Literal["measured", "not_applicable", "no_eligible_cases"]
EvaluationResultKind = Literal["standard_benchmark", "custom_evaluation"]
EvaluationSummary = Mapping[str, object]
RETRIEVAL_SYSTEMS: tuple[SystemName, ...] = ("dense", "bm25", "hybrid")
FIXED_RAG_SYSTEMS: tuple[SystemName, ...] = ("dense-rag", "bm25-rag", "hybrid-rag")
FULL_RAG_SYSTEM: SystemName = "full-rag"
ANSWER_SYSTEMS: tuple[SystemName, ...] = (*FIXED_RAG_SYSTEMS, FULL_RAG_SYSTEM)
SYSTEMS: tuple[SystemName, ...] = (*RETRIEVAL_SYSTEMS, *ANSWER_SYSTEMS)
STANDARD_BENCHMARK_DATASET = "multihop"
STANDARD_BENCHMARK_SPLIT: Split = "development"
CANONICAL_BENCHMARK_CASE_IDS = (
    "multihop-0063",
    "multihop-0716",
    "multihop-0994",
    "multihop-2512",
    "multihop-0460",
    "multihop-0457",
    "multihop-1441",
    "multihop-0590",
    "multihop-1576",
    "multihop-1651",
    "multihop-1673",
    "multihop-1946",
    "multihop-1233",
    "multihop-1798",
    "multihop-2477",
    "multihop-1778",
    "multihop-2436",
    "multihop-0842",
    "multihop-1772",
    "multihop-0685",
)
CANONICAL_BENCHMARK_RESULT_COUNT = len(CANONICAL_BENCHMARK_CASE_IDS) * len(SYSTEMS)
CANONICAL_CHAT_MODEL = "qwen3.5:9b"
CANONICAL_EMBEDDING_MODEL = "nomic-embed-text"
CANONICAL_TEMPERATURE = 0.0
CANONICAL_CHUNK_SIZE = 700
CANONICAL_CHUNK_OVERLAP = 100
CANONICAL_RETRIEVAL_LIMIT = 5
CANONICAL_SEMANTIC_CANDIDATES = 10
CANONICAL_SPARSE_CANDIDATES = 10
CANONICAL_MAX_CONTEXT_CHUNKS = 6
CANONICAL_RETRY_LIMIT = 1
CANONICAL_SUBQUERY_LIMIT = 4
CANONICAL_REQUEST_TIMEOUT_SECONDS = 30.0
FIXED_RAG_PROMPT_ID = "fixed_rag_grounded_answer"
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


def _has_canonical_configuration(configuration: object) -> bool:
    if not isinstance(configuration, dict):
        return False
    systems = configuration.get("systems")
    if not isinstance(systems, list):
        return False
    expected_configuration = {
        "dataset_name": STANDARD_BENCHMARK_DATASET,
        "evaluated_split": STANDARD_BENCHMARK_SPLIT,
        "chat_model": CANONICAL_CHAT_MODEL,
        "embedding_model": CANONICAL_EMBEDDING_MODEL,
        "temperature": CANONICAL_TEMPERATURE,
        "fixed_rag_prompt_id": FIXED_RAG_PROMPT_ID,
        "chunk_size": CANONICAL_CHUNK_SIZE,
        "chunk_overlap": CANONICAL_CHUNK_OVERLAP,
        "retrieval_limit": CANONICAL_RETRIEVAL_LIMIT,
        "semantic_candidates": CANONICAL_SEMANTIC_CANDIDATES,
        "sparse_candidates": CANONICAL_SPARSE_CANDIDATES,
        "retry_limit": CANONICAL_RETRY_LIMIT,
        "subquery_limit": CANONICAL_SUBQUERY_LIMIT,
        "case_timeout_seconds": CANONICAL_REQUEST_TIMEOUT_SECONDS,
    }
    return (
        len(systems) == len(SYSTEMS)
        and set(systems) == set(SYSTEMS)
        and all(
            configuration.get(name) == expected for name, expected in expected_configuration.items()
        )
    )


def _has_canonical_results(results: Sequence[CaseResult]) -> bool:
    expected_pairs = {
        (case_id, system) for case_id in CANONICAL_BENCHMARK_CASE_IDS for system in SYSTEMS
    }
    actual_pairs = {(item.case_id, item.system) for item in results}
    return (
        len(results) == CANONICAL_BENCHMARK_RESULT_COUNT
        and len(actual_pairs) == CANONICAL_BENCHMARK_RESULT_COUNT
        and actual_pairs == expected_pairs
    )


def is_complete_full_rag_benchmark_artifact(
    *,
    summary: EvaluationSummary,
    results: Sequence[CaseResult],
) -> bool:
    """Return whether an artifact contains the complete Full RAG Benchmark."""
    case_ids = summary.get("case_ids")
    if not isinstance(case_ids, list) or not all(isinstance(case_id, str) for case_id in case_ids):
        return False
    return (
        summary.get("benchmark_name") == "full_rag_benchmark"
        and _has_canonical_configuration(summary.get("configuration"))
        and len(case_ids) == len(CANONICAL_BENCHMARK_CASE_IDS)
        and set(case_ids) == set(CANONICAL_BENCHMARK_CASE_IDS)
        and summary.get("expected_result_count") == CANONICAL_BENCHMARK_RESULT_COUNT
        and summary.get("completed_result_count") == CANONICAL_BENCHMARK_RESULT_COUNT
        and _has_canonical_results(results)
    )


def evaluation_result_kind(
    *,
    summary: EvaluationSummary,
    results: Sequence[CaseResult],
) -> EvaluationResultKind:
    """Classify complete Full RAG Benchmark artifacts and custom evaluations."""
    return (
        "standard_benchmark"
        if is_complete_full_rag_benchmark_artifact(summary=summary, results=results)
        else "custom_evaluation"
    )


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


class EvaluationSubquery(BaseModel):
    id: str
    text: str


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
    evidence_status: str | None = None
    subquery_specs: list[EvaluationSubquery] = Field(default_factory=list)
    rewritten_subqueries: list[EvaluationSubquery] = Field(default_factory=list)
    supported_subquery_ids: list[str] = Field(default_factory=list)
    relevant_labels: list[str] = Field(default_factory=list)
    termination_reason: str | None = None
    validation_violations: list[str] = Field(default_factory=list)
    initial_validation_violations: list[str] = Field(default_factory=list)
    repair_validation_violations: list[str] = Field(default_factory=list)
    retrieved_evidence: list[dict[str, JsonValue]] = Field(default_factory=list)
    public_trace: list[dict[str, JsonValue]] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    run_id: str
    timestamp: str
    git_commit: str
    dataset_hash: str
    evaluated_split: Split
    systems: list[SystemName]
    chat_model: str
    embedding_model: str
    temperature: float = CANONICAL_TEMPERATURE
    fixed_rag_prompt_id: str = FIXED_RAG_PROMPT_ID
    chunk_size: int
    chunk_overlap: int
    retrieval_limit: int
    semantic_candidates: int
    sparse_candidates: int
    retry_limit: int
    subquery_limit: int
    case_timeout_seconds: float = Field(default=30.0, gt=0)
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
