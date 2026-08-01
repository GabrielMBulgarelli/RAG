"""Presentation-neutral application data contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

RuntimeState = Literal["ready", "not_loaded", "blocked", "error"]
CheckState = Literal["ready", "review", "blocked", "not_loaded", "error"]
CorpusState = Literal["ready", "empty", "review", "error"]
DiagnosticsState = Literal["ready", "review", "blocked", "error"]
ConversationRole = Literal["user", "assistant"]
AnswerState = Literal["supported", "limited", "abstention", "unavailable", "completed"]


class OperationKind(str, Enum):
    INDEX_DOCUMENTS = "index_documents"
    DELETE_DOCUMENT = "delete_document"
    LOAD_MODEL = "load_model"
    QUERY = "query"
    BENCHMARK = "benchmark"


class ActiveOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_id: UUID
    kind: OperationKind
    started_at: datetime
    benchmark_run_id: UUID | None = None
    cancellation_requested: bool = False


class CapabilitySnapshot(BaseModel):
    can_query: bool
    can_load_models: bool
    can_upload: bool
    can_run_benchmark: bool


class CorpusSnapshot(BaseModel):
    document_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    status: CorpusState


class RuntimeSnapshot(BaseModel):
    state: RuntimeState
    configured_chat_model: str
    active_chat_model: str | None
    embedding_model: str
    available_chat_models: list[str]
    detail: str
    capabilities: CapabilitySnapshot
    active_operation: ActiveOperation | None
    corpus: CorpusSnapshot


class DiagnosticCheck(BaseModel):
    area: str
    name: str
    state: CheckState
    detail: str


class DiagnosticsSnapshot(BaseModel):
    state: DiagnosticsState
    title: str
    detail: str
    runtime_checks: list[DiagnosticCheck]
    index_checks: list[DiagnosticCheck]
    evaluation_checks: list[DiagnosticCheck]
    active_operation: ActiveOperation | None
    stale: bool


class DocumentRecord(BaseModel):
    id: str
    filename: str
    state: str
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    indexed_at: datetime | None
    updated_at: datetime

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must not be empty")
        return value


class DocumentList(BaseModel):
    documents: list[DocumentRecord]
    corpus: CorpusSnapshot
    active_operation: ActiveOperation | None


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content_type: str | None
    content: bytes


class UploadAccepted(BaseModel):
    filename: str
    document_id: str

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document_id must not be empty")
        return value


class UploadBatchResult(BaseModel):
    accepted: list[UploadAccepted]
    documents: list[DocumentRecord]
    corpus: CorpusSnapshot


class ConversationMessage(BaseModel):
    id: UUID
    role: ConversationRole
    content: str
    created_at: datetime


class Source(BaseModel):
    label: str
    filename: str
    page: int | None
    excerpt: str


class RetrievalHit(BaseModel):
    chunk_id: str
    filename: str
    page: int | None
    semantic_score: float | None
    sparse_score: float | None
    fused_score: float | None
    selection_score: float | None
    matched_subqueries: list[str]


class TraceEvent(BaseModel):
    stage: str
    decision: str
    retrieved_count: int | None = Field(ge=0)
    fused_count: int | None = Field(ge=0)
    selected_count: int | None = Field(ge=0)
    retry_count: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    termination: str
    duration_ms: float | None = Field(ge=0)


class QueryDiagnostics(BaseModel):
    route: str
    retrieval_strategy: str
    subqueries: list[str]
    retry_count: int = Field(ge=0)
    evidence_state: str
    conflict_state: str
    citation_validation: str


class QueryRequest(BaseModel):
    session_id: UUID
    question: str

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("question must not be empty")
        return trimmed


class QueryResponse(BaseModel):
    session_id: UUID
    message: ConversationMessage
    answer_state: AnswerState
    sources: list[Source]
    retrieval_hits: list[RetrievalHit]
    trace: list[TraceEvent]
    diagnostics: QueryDiagnostics


class ModelLoadRequest(BaseModel):
    chat_model: str

    @field_validator("chat_model")
    @classmethod
    def trim_chat_model(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("chat_model must not be empty")
        return trimmed


class ConversationExportRequest(BaseModel):
    session_id: UUID


class ApiProblem(BaseModel):
    code: str
    message: str
    details: dict[str, JsonValue]


class ResourceLinks(BaseModel):
    run: str
    events: str
    download: str


class BenchmarkRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkStartResponse(BaseModel):
    run_id: UUID
    status: BenchmarkRunStatus
    links: ResourceLinks


class BenchmarkMetricStatus(str, Enum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    NO_ELIGIBLE_CASES = "no_eligible_cases"


class BenchmarkMetricObservation(BaseModel):
    system: str
    value: float | None
    status: BenchmarkMetricStatus
    sample_count: int = Field(ge=0)
    note: str | None = None

    @model_validator(mode="after")
    def validate_value_matches_status(self) -> Self:
        if self.status is BenchmarkMetricStatus.MEASURED and self.value is None:
            raise ValueError("measured metrics require a value")
        if self.status is not BenchmarkMetricStatus.MEASURED and self.value is not None:
            raise ValueError("unmeasured metrics must use a null value")
        return self


class BenchmarkCaseMetricObservation(BenchmarkMetricObservation):
    name: str
    label: str


class BenchmarkCaseOutcome(str, Enum):
    SUCCESSFUL = "successful"
    EXPECTATION_FAILURE = "expectation_failure"
    RUNTIME_FAILURE = "runtime_failure"


class BenchmarkCaseSummary(BaseModel):
    case_id: str
    system: str
    question: str
    outcome: BenchmarkCaseOutcome
    failure_classification: str | None


class BenchmarkMetric(BaseModel):
    name: str
    label: str
    observations: list[BenchmarkMetricObservation]


class BenchmarkSystem(BaseModel):
    id: str
    label: str


class BenchmarkSection(BaseModel):
    id: str
    title: str
    system_ids: list[str] = Field(default_factory=list)
    metrics: list[BenchmarkMetric]
    detail: str | None = None


class BenchmarkFailure(BaseModel):
    case_id: str
    system: str
    classification: str
    detail: str


class BenchmarkProgress(BaseModel):
    completed_cases: int = Field(ge=0)
    total_cases: int = Field(ge=0)
    current_system: str | None = None
    current_system_index: int | None = Field(default=None, ge=0)
    total_systems: int = Field(default=0, ge=0)
    current_case_id: str | None = None
    current_case_index: int | None = Field(default=None, ge=0)


class BenchmarkGraphConfiguration(BaseModel):
    max_candidates: int = Field(ge=1)
    maximum_context_chunks: int = Field(ge=1)
    retry_limit: int = Field(ge=0)
    subquery_limit: int = Field(ge=1)


class BenchmarkReproducibility(BaseModel):
    benchmark_name: str
    git_commit: str
    dataset_identifier: str
    dataset_hash: str
    case_ids: list[str]
    expected_result_count: int = Field(ge=0)
    chat_model: str
    embedding_model: str
    temperature: float
    fixed_rag_prompt_id: str
    graph_configuration: BenchmarkGraphConfiguration
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    retrieval_limit: int = Field(ge=1)
    semantic_candidates: int = Field(ge=1)
    sparse_candidates: int = Field(ge=1)
    maximum_context_chunks: int = Field(ge=1)
    retry_limit: int = Field(ge=0)
    subquery_limit: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)


class BenchmarkMetadata(BaseModel):
    dataset: str
    split: str
    systems: list[BenchmarkSystem]
    chat_model: str
    embedding_model: str
    started_at: datetime | None
    completed_at: datetime | None
    reproducibility: dict[str, JsonValue] = Field(default_factory=dict)


class BenchmarkRun(BaseModel):
    run_id: UUID
    status: BenchmarkRunStatus
    progress: BenchmarkProgress
    metadata: BenchmarkMetadata
    sections: list[BenchmarkSection]
    failures: list[BenchmarkFailure]
    links: ResourceLinks
    error: ApiProblem | None


class BenchmarkEventType(str, Enum):
    BENCHMARK_STARTED = "benchmark.started"
    SYSTEM_STARTED = "system.started"
    CASE_STARTED = "case.started"
    CASE_COMPLETED = "case.completed"
    CASE_FAILED = "case.failed"
    SYSTEM_COMPLETED = "system.completed"
    BENCHMARK_CANCELLATION_REQUESTED = "benchmark.cancellation_requested"
    BENCHMARK_CANCELLED = "benchmark.cancelled"
    BENCHMARK_COMPLETED = "benchmark.completed"
    BENCHMARK_FAILED = "benchmark.failed"
    HEARTBEAT = "heartbeat"


class BenchmarkEvent(BaseModel):
    event_id: int = Field(ge=1)
    run_id: UUID
    type: BenchmarkEventType
    timestamp: datetime
    data: dict[str, JsonValue]


class BenchmarkCaseDetail(BaseModel):
    case_id: str
    system: str
    question: str
    expected_answer: str | None
    generated_answer: str | None
    expected_evidence: list[dict[str, JsonValue]]
    retrieved_evidence: list[dict[str, JsonValue]]
    metric_observations: list[BenchmarkCaseMetricObservation]
    failure_classification: str | None
    public_trace: list[TraceEvent]
    sanitized_raw_result: dict[str, JsonValue] | None
