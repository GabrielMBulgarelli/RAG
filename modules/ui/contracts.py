"""Immutable, presentation-neutral records exposed by the UI controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RuntimeState = Literal["ready", "not_loaded", "blocked", "error"]
CheckState = Literal["ready", "review", "blocked", "not_loaded", "error"]
CorpusState = Literal["ready", "empty", "review", "error"]
IndexState = Literal["ready", "review", "error"]
AnswerState = Literal["supported", "limited", "abstention", "unavailable", "completed"]
AttentionLevel = Literal["info", "warning", "error"]
EvaluationPageState = Literal["ready", "blocked", "saved_result", "error"]
SystemPageState = Literal["ready", "review", "blocked", "error"]


@dataclass(frozen=True)
class SystemCheck:
    area: str
    name: str
    state: CheckState
    detail: str


@dataclass(frozen=True)
class SafeConfigurationValue:
    name: str
    value: str


@dataclass(frozen=True)
class SystemPageSnapshot:
    state: SystemPageState
    title: str
    detail: str
    can_load_models: bool
    runtime_checks: tuple[SystemCheck, ...]
    index_checks: tuple[SystemCheck, ...]
    evaluation_checks: tuple[SystemCheck, ...]
    safe_configuration: tuple[SafeConfigurationValue, ...]


@dataclass(frozen=True)
class RuntimeSnapshot:
    state: RuntimeState
    title: str
    detail: str
    chat_enabled: bool
    can_load_models: bool
    checks: tuple[SystemCheck, ...]
    chat_model: str = ""
    embedding_model: str = ""


@dataclass(frozen=True)
class CorpusSnapshot:
    document_count: int
    page_count: int
    chunk_count: int
    status: CorpusState


@dataclass(frozen=True)
class IndexSnapshot:
    missing_chunk_count: int
    orphan_chunk_count: int
    duplicate_id_count: int
    missing_source_file_count: int
    incompatible_document_count: int
    status: IndexState


@dataclass(frozen=True)
class SourceView:
    label: str
    filename: str
    page: int | None
    excerpt: str


@dataclass(frozen=True)
class RetrievalHitView:
    chunk_id: str
    filename: str
    page: int | None
    semantic_score: float | None
    sparse_score: float | None
    fused_score: float | None
    selection_score: float | None
    matched_subqueries: tuple[str, ...]


@dataclass(frozen=True)
class TraceEventView:
    stage: str
    decision: str
    retrieved_count: int | None
    fused_count: int | None
    selected_count: int | None
    retry_count: int
    llm_calls: int
    termination: str
    duration_ms: float | None


@dataclass(frozen=True)
class QueryDiagnostics:
    route: str
    retrieval_strategy: str
    subqueries: tuple[str, ...]
    retry_count: int
    evidence_state: str
    conflict_state: str
    citation_validation: str


@dataclass(frozen=True)
class QuerySnapshot:
    answer: str
    answer_state: AnswerState
    sources: tuple[SourceView, ...]
    retrieval_hits: tuple[RetrievalHitView, ...]
    trace: tuple[TraceEventView, ...]
    diagnostics: QueryDiagnostics


@dataclass(frozen=True)
class EvaluationSummary:
    result_path: str
    split: str
    systems: tuple[str, ...]
    case_count: int
    result_kind: Literal["standard", "custom"]
    created_at: str
    chat_model: str = "—"
    quality_categories: tuple["EvaluationCategorySummary", ...] = ()


@dataclass(frozen=True)
class EvaluationMetricObservation:
    system: str
    value: float | None
    status: str
    sample_count: int | None


@dataclass(frozen=True)
class EvaluationMetricSummary:
    name: str
    label: str
    observations: tuple[EvaluationMetricObservation, ...]


@dataclass(frozen=True)
class EvaluationCategorySummary:
    name: str
    metrics: tuple[EvaluationMetricSummary, ...]


@dataclass(frozen=True)
class EvaluationPageSnapshot:
    state: EvaluationPageState
    split: str
    systems: tuple[str, ...]
    requires_index: bool
    requires_embeddings: bool
    requires_chat: bool
    problems: tuple[str, ...]
    latest: EvaluationSummary | None
    chat_model: str = "—"
    metric_rows: tuple[tuple[str, ...], ...] = ()
    failure_rows: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class AttentionItem:
    area: str
    message: str
    level: AttentionLevel


@dataclass(frozen=True)
class DashboardSnapshot:
    runtime: RuntimeSnapshot
    corpus: CorpusSnapshot
    index: IndexSnapshot
    evaluation: EvaluationSummary | None
    attention_items: tuple[AttentionItem, ...]
