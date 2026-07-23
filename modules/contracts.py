"""Structural contracts shared across application and retrieval boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

from langchain_core.documents import Document

from .models import IngestionManifest, IngestionResult, ReconciliationResult

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type Metadata = dict[str, JsonValue]
type VectorFilter = dict[str, JsonValue]
type TableCell = str | int | float | bool | None
type TableRow = list[TableCell]


class VectorGetResult(TypedDict, total=False):
    """The subset of vector-store payload fields used by retrieval."""

    ids: list[str]
    documents: list[str] | None
    metadatas: list[Metadata | None] | None


@runtime_checkable
class RetrievalVectorStore(Protocol):
    """Operations required by hybrid retrieval."""

    def get(
        self,
        *,
        include: list[str],
        where: VectorFilter | None = None,
    ) -> VectorGetResult: ...

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        *,
        k: int,
        filter: VectorFilter | None = None,
    ) -> list[tuple[Document, float]]: ...


class SourceSettings(Protocol):
    """Configuration exposed to application-level document workflows."""

    @property
    def sources_dir(self) -> Path: ...


@runtime_checkable
class ApplicationVectorStore(Protocol):
    """Vector-storage operations consumed by the application facade."""

    @property
    def settings(self) -> SourceSettings: ...

    delete_document: Callable[[str], bool]

    def setup(self) -> object: ...

    def manifest(self) -> IngestionManifest: ...

    def save_uploads(self, paths: list[str] | None) -> list[Path]: ...

    def index_document(self, path: str | Path) -> IngestionResult: ...

    def rebuild(self) -> int: ...

    def reconcile_index(self) -> ReconciliationResult: ...

    def chunk_count(self) -> int: ...


class GraphVectorStore(Protocol):
    """Vector-storage operations required while composing the RAG graph."""

    def setup(self) -> object: ...

    def document_names(self) -> list[str]: ...
