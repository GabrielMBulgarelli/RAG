"""Small data contracts shared by the MVP workflow."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Route(StrEnum):
    CATALOG = "catalog"
    CLARIFICATION = "clarification"
    OUT_OF_SCOPE = "out_of_scope"
    SIMPLE_SEARCH = "simple_search"
    COMPLEX_SEARCH = "complex_search"


class RetrievalStrategy(StrEnum):
    NONE = "none"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class EvidenceStatus(StrEnum):
    SUFFICIENT = "sufficient"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"


class AnswerViolation(StrEnum):
    EMPTY_ANSWER = "empty_answer"
    CITATIONS_ONLY = "citations_only"
    UNKNOWN_LABEL = "unknown_label"
    IRRELEVANT_CITATION = "irrelevant_citation"
    UNCITED_CLAIM = "uncited_claim"


class RouteDecision(BaseModel):
    route: Route
    strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC
    reason: str = ""


class QueryDecomposition(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=4)


class SubquerySpec(BaseModel):
    id: str
    text: str = Field(min_length=1)


class QueryRefinement(BaseModel):
    rewrites: dict[str, str] = Field(default_factory=dict)


class SubqueryEvidence(BaseModel):
    subquery_id: str
    relevant_labels: list[str] = Field(default_factory=list)


class EvidenceDecision(BaseModel):
    answer_supported: bool
    drafted_answer: str = ""
    assessments: list[SubqueryEvidence] = Field(default_factory=list)
    conflict: bool = False
    conflicting_labels: list[str] = Field(default_factory=list)
    reason: str = ""


class EvidenceGrade(BaseModel):
    status: EvidenceStatus
    answer_supported: bool
    drafted_answer: str = ""
    assessments: list[SubqueryEvidence] = Field(default_factory=list)
    relevant_labels: list[str] = Field(default_factory=list)
    supported_subqueries: list[str] = Field(default_factory=list)
    unsupported_subqueries: list[str] = Field(default_factory=list)
    relevant_labels_by_subquery: dict[str, list[str]] = Field(default_factory=dict)
    coverage_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    fully_supported: bool = False
    partially_supported: bool = False
    conflict: bool = False
    conflicting_labels: list[str] = Field(default_factory=list)
    reason: str = ""


class RetrievalHit(BaseModel):
    chunk_id: str
    content: str
    document_id: str = ""
    filename: str
    page: int
    score: float
    semantic_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    selection_score: float | None = None
    subqueries: list[str] = Field(default_factory=list)


class RetrievalBatch(BaseModel):
    hits: list[RetrievalHit] = Field(default_factory=list)
    retrieved_count: int = Field(ge=0)
    fused_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)


class TraceEvent(BaseModel):
    stage: str
    decision: str | None = None
    candidate_count: int | None = Field(default=None, ge=0)
    retrieved_count: int | None = Field(default=None, ge=0)
    fused_count: int | None = Field(default=None, ge=0)
    selected_count: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0, le=1)
    llm_calls: int = Field(default=0, ge=0)
    duration_ms: float = Field(ge=0)
    termination: str | None = None


class CitationSource(BaseModel):
    label: str
    chunk_id: str
    filename: str
    page: int
    excerpt: str


class AnswerValidation(BaseModel):
    sanitized_text: str
    used_sources: list[CitationSource] = Field(default_factory=list)
    violations: list[AnswerViolation] = Field(default_factory=list)
    unknown_labels: list[str] = Field(default_factory=list)
    irrelevant_labels: list[str] = Field(default_factory=list)
    uncited_claims: list[str] = Field(default_factory=list)
    empty_answer: bool = False
    citations_only: bool = False
    is_valid: bool = True
    repair_attempted: bool = False
    repair_succeeded: bool = False
    initial_violations: list[AnswerViolation] = Field(default_factory=list)
    repair_violations: list[AnswerViolation] = Field(default_factory=list)


class RAGResult(BaseModel):
    answer: str
    standalone_query: str = ""
    route: Route
    strategy: RetrievalStrategy
    retry_count: int = Field(ge=0, le=1)
    evidence_status: EvidenceStatus
    sources: list[CitationSource] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list, max_length=4)
    subquery_specs: list[SubquerySpec] = Field(default_factory=list, max_length=4)
    rewritten_subqueries: list[SubquerySpec] = Field(default_factory=list, max_length=4)
    supported_subquery_ids: list[str] = Field(default_factory=list)
    relevant_labels: list[str] = Field(default_factory=list)
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    validation: AnswerValidation = Field(
        default_factory=lambda: AnswerValidation(sanitized_text="")
    )
    conflict: bool = False


class ManifestDocument(BaseModel):
    document_id: str
    relative_path: str
    filename: str
    content_hash: str
    chunk_ids: list[str]
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    embedding_model: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    size_bytes: int = Field(default=0, ge=0)
    indexed_at: datetime | None = None
    updated_at: datetime | None = None


class IngestionManifest(BaseModel):
    schema_version: int = 2
    documents: dict[str, ManifestDocument] = Field(default_factory=dict)


class IngestionError(BaseModel):
    document: str
    operation: str
    error_type: str
    message: str


class IngestionResult(BaseModel):
    document_id: str
    success: bool
    chunk_count: int = 0
    error: IngestionError | None = None


class ReconciliationResult(BaseModel):
    missing_chunk_ids: list[str] = Field(default_factory=list)
    orphan_chunk_ids: list[str] = Field(default_factory=list)
    duplicate_chunk_ids: list[str] = Field(default_factory=list)
    missing_source_files: list[Path] = Field(default_factory=list)
    incompatible_document_ids: list[str] = Field(default_factory=list)
