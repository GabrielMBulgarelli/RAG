"""Ingestion transactions and maintenance operations for the vector store."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from .config import Settings
from .models import (
    IngestionError,
    IngestionManifest,
    IngestionResult,
    ManifestDocument,
    ReconciliationResult,
)


class _VectorContext(Protocol):
    settings: Settings
    embeddings: Embeddings
    vectorstore: Chroma | None

    def document_id(self, path: str | Path) -> str: ...

    def manifest(self) -> IngestionManifest: ...

    def _write_manifest(self, manifest: IngestionManifest) -> None: ...

    def load_documents(self, paths: Sequence[str | Path] | None = None) -> list[Document]: ...

    def prepare_chunks(self, documents: list[Document]) -> list[Document]: ...

    def _relative_path(self, path: Path) -> str: ...

    def _store(self) -> Chroma: ...

    def index_document(self, path: str | Path) -> IngestionResult: ...

    def index_documents(self, paths: Sequence[str | Path] | None = None) -> int: ...

    def rebuild(self) -> int: ...


def _stored_chunks(store: Chroma, *, chunk_ids: set[str]) -> list[Document]:
    if not chunk_ids:
        return []
    payload = store.get(ids=sorted(chunk_ids), include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(
            payload.get("documents") or [], payload.get("metadatas") or [], strict=True
        )
    ]


@dataclass(frozen=True)
class _ManifestUpdate:
    previous: IngestionManifest
    source: Path
    chunks: list[Document]
    chunk_ids: list[str]


def _updated_manifest(context: _VectorContext, *, request: _ManifestUpdate) -> IngestionManifest:
    document_id = context.document_id(request.source)
    updated = request.previous.model_copy(deep=True)
    updated.documents[document_id] = ManifestDocument(
        document_id=document_id,
        relative_path=context._relative_path(request.source),
        filename=request.source.name,
        content_hash=hashlib.sha256(request.source.read_bytes()).hexdigest(),
        chunk_ids=request.chunk_ids,
        page_count=len({int(chunk.metadata.get("page", 1)) for chunk in request.chunks}),
        chunk_count=len(request.chunk_ids),
        embedding_model=context.settings.embedding_model,
        chunk_size=context.settings.chunk_size,
        chunk_overlap=context.settings.chunk_overlap,
    )
    return updated


def _restore_previous(
    context: _VectorContext,
    *,
    inserted_ids: set[str],
    previous_chunks: list[Document],
) -> None:
    if context.vectorstore is None:
        return
    if inserted_ids:
        context.vectorstore.delete(ids=sorted(inserted_ids))
    if previous_chunks:
        context.vectorstore.add_documents(
            previous_chunks,
            ids=[str(chunk.metadata["chunk_id"]) for chunk in previous_chunks],
        )


class VectorIngestionMixin:
    def _store(self: _VectorContext) -> Chroma:
        if self.vectorstore is None:
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name="rag_documents",
                persist_directory=str(self.settings.chroma_dir),
                embedding_function=self.embeddings,
            )
        return self.vectorstore

    def index_documents(self: _VectorContext, paths: Sequence[str | Path] | None = None) -> int:
        selected = paths
        if selected is None:
            self.settings.sources_dir.mkdir(parents=True, exist_ok=True)
            selected = sorted(
                (
                    *self.settings.sources_dir.rglob("*.pdf"),
                    *self.settings.sources_dir.rglob("*.txt"),
                )
            )
        return sum(
            result.chunk_count
            for result in (self.index_document(Path(path)) for path in selected)
            if result.success
        )

    def index_document(self: _VectorContext, path: str | Path) -> IngestionResult:
        source = Path(path)
        document_id = self.document_id(source)
        manifest = self.manifest()
        previous_record = manifest.documents.get(document_id)
        old_ids = set(previous_record.chunk_ids) if previous_record else set()
        inserted_ids: set[str] = set()
        previous_chunks: list[Document] = []
        try:
            chunks = self.prepare_chunks(self.load_documents([source]))
            if not chunks:
                raise ValueError("No supported content was parsed")
            ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            store = self._store()
            previous_chunks = _stored_chunks(store, chunk_ids=old_ids)
            inserted_ids = set(ids) - old_ids
            store.add_documents(chunks, ids=ids)
            if set(store.get(ids=ids)["ids"]) != set(ids):
                raise RuntimeError("New chunks could not be verified after upsert")
            stale_ids = old_ids - set(ids)
            if stale_ids:
                store.delete(ids=sorted(stale_ids))
            self._write_manifest(
                _updated_manifest(
                    self,
                    request=_ManifestUpdate(
                        previous=manifest, source=source, chunks=chunks, chunk_ids=ids
                    ),
                )
            )
            return IngestionResult(document_id=document_id, success=True, chunk_count=len(ids))
        except Exception as exc:
            _restore_previous(self, inserted_ids=inserted_ids, previous_chunks=previous_chunks)
            return IngestionResult(
                document_id=document_id,
                success=False,
                error=IngestionError(
                    document=source.name,
                    operation="index",
                    error_type=type(exc).__name__,
                    message=str(exc),
                ),
            )

    def setup(self: _VectorContext, force_rebuild: bool = False) -> Chroma:
        if force_rebuild:
            self.rebuild()
        return self._store()

    def get_retriever(
        self: _VectorContext, search_kwargs: dict[str, object] | None = None
    ) -> VectorStoreRetriever:
        return self._store().as_retriever(
            search_kwargs=search_kwargs or {"k": self.settings.k_retrieval}
        )


class VectorMaintenanceMixin:
    def save_uploads(self: _VectorContext, paths: list[str] | None) -> list[Path]:
        self.settings.sources_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for raw_path in paths or []:
            source = Path(raw_path)
            if source.suffix.lower() not in {".pdf", ".txt"}:
                continue
            content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            upload_key = hashlib.sha256(f"{source.name}\0{content_hash}".encode()).hexdigest()
            target = self.settings.sources_dir / upload_key / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            saved.append(target)
        return saved

    def rebuild(self: _VectorContext) -> int:
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
        elif self.settings.chroma_dir.exists():
            shutil.rmtree(self.settings.chroma_dir)
        if self.settings.manifest_path.exists():
            self.settings.manifest_path.unlink()
        return self.index_documents()

    def has_deleted_document(
        self: _VectorContext, document_id: str, *, delete_source: bool = True
    ) -> bool:
        manifest = self.manifest()
        record = manifest.documents.get(document_id)
        if record is None:
            return False
        if record.chunk_ids:
            self._store().delete(ids=record.chunk_ids)
        del manifest.documents[document_id]
        self._write_manifest(manifest)
        if delete_source:
            source = self.settings.sources_dir / record.relative_path
            if source.exists():
                source.unlink()
                if source.parent != self.settings.sources_dir:
                    try:
                        source.parent.rmdir()
                    except OSError:
                        pass
        return True

    delete_document = has_deleted_document

    def reconcile_index(self: _VectorContext) -> ReconciliationResult:
        manifest = self.manifest()
        stored_ids = [str(item) for item in self._store().get()["ids"]]
        expected_ids = {
            chunk_id for document in manifest.documents.values() for chunk_id in document.chunk_ids
        }
        incompatible = [
            document.document_id
            for document in manifest.documents.values()
            if document.embedding_model != self.settings.embedding_model
            or document.chunk_size != self.settings.chunk_size
            or document.chunk_overlap != self.settings.chunk_overlap
        ]
        missing_sources = [
            Path(document.relative_path)
            for document in manifest.documents.values()
            if not (self.settings.sources_dir / document.relative_path).exists()
        ]
        duplicates = sorted({item for item in stored_ids if stored_ids.count(item) > 1})
        return ReconciliationResult(
            missing_chunk_ids=sorted(expected_ids - set(stored_ids)),
            orphan_chunk_ids=sorted(set(stored_ids) - expected_ids),
            duplicate_chunk_ids=duplicates,
            missing_source_files=missing_sources,
            incompatible_document_ids=sorted(incompatible),
        )

    def document_names(self: _VectorContext) -> list[str]:
        stored = self._store().get(include=["metadatas"])
        return sorted(
            {str(item.get("filename", "Unknown")) for item in stored.get("metadatas") or []}
        )

    def chunk_count(self: _VectorContext) -> int:
        return int(self._store()._collection.count())
