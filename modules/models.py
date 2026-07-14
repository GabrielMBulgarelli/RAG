"""Small data contracts shared by the MVP workflow."""

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


class RouteDecision(BaseModel):
    route: Route
    strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC
    reason: str = ""


class QueryDecomposition(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=4)


class EvidenceGrade(BaseModel):
    status: EvidenceStatus
    relevant_labels: list[str] = Field(default_factory=list)
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
    subqueries: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    stage: str
    decision: str | None = None
    candidate_count: int | None = Field(default=None, ge=0)
    selected_count: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0, le=1)
    duration_ms: float = Field(ge=0)
    termination: str | None = None


class CitationSource(BaseModel):
    label: str
    chunk_id: str
    filename: str
    page: int
    excerpt: str


class RAGResult(BaseModel):
    answer: str
    standalone_query: str = ""
    route: Route
    strategy: RetrievalStrategy
    retry_count: int = Field(ge=0, le=1)
    evidence_status: EvidenceStatus
    sources: list[CitationSource] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list, max_length=4)
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)


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


class IngestionManifest(BaseModel):
    schema_version: int = 1
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
