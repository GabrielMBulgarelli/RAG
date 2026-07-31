"""Evaluation schemas and benchmark contracts."""

from __future__ import annotations

from typing import Any, Literal, Self

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
RETRIEVAL_SYSTEMS: tuple[SystemName, ...] = ("dense", "bm25", "hybrid")
FIXED_RAG_SYSTEMS: tuple[SystemName, ...] = ("dense-rag", "bm25-rag", "hybrid-rag")
FULL_RAG_SYSTEM: SystemName = "full-rag"
ANSWER_SYSTEMS: tuple[SystemName, ...] = (*FIXED_RAG_SYSTEMS, FULL_RAG_SYSTEM)
SYSTEMS: tuple[SystemName, ...] = (*RETRIEVAL_SYSTEMS, *ANSWER_SYSTEMS)
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


def is_complete_full_rag_benchmark_artifact(summary: dict[str, Any]) -> bool:
    """Return whether an artifact contains the complete Full RAG Benchmark."""
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
    """Classify complete Full RAG Benchmark artifacts and custom evaluations."""
    return (
        "standard_benchmark"
        if is_complete_full_rag_benchmark_artifact(summary)
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
