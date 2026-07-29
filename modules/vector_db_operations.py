"""Ingestion transactions and maintenance operations for the vector store."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from pypdf import PdfReader

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

    def index_upload_batch_with_manifest(
        self, uploads: Sequence[_UploadedFile]
    ) -> tuple[list[ManifestDocument], IngestionManifest]: ...

    def delete_document_with_manifest(
        self, document_id: str, *, delete_source: bool = True
    ) -> IngestionManifest | None: ...

    def rebuild(self) -> int: ...


class _UploadedFile(Protocol):
    @property
    def filename(self) -> str: ...

    @property
    def content_type(self) -> str | None: ...

    @property
    def content(self) -> bytes: ...


class UploadValidationError(ValueError):
    pass


class UploadLimitExceededError(ValueError):
    pass


class VectorTransactionError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ManifestUpdate:
    previous: IngestionManifest
    source: Path
    chunks: list[Document]
    chunk_ids: list[str]


@dataclass(frozen=True)
class _PreparedUpload:
    filename: str
    content: bytes
    target: Path
    relative_path: str
    document_id: str


_MAX_UPLOAD_FILES = 10
_MAX_FILE_BYTES = 25 * 1024 * 1024
_MAX_TOTAL_BYTES = 100 * 1024 * 1024
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "CON",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
    "NUL",
    "PRN",
}


def _safe_basename(submitted: str) -> str:
    basename = submitted.replace("\\", "/").rsplit("/", 1)[-1]
    sanitized = re.sub(r"[^\w .-]", "_", basename).strip(" .")
    if not sanitized:
        raise UploadValidationError("Uploaded filename is invalid.")
    if Path(sanitized).stem.upper() in _WINDOWS_RESERVED_NAMES:
        sanitized = f"_{sanitized}"
    return sanitized


def _validate_upload_content(filename: str, content: bytes) -> None:
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise UploadValidationError("Only PDF and TXT uploads are supported.")
    if not content:
        raise UploadValidationError("Uploaded files must not be empty.")
    if extension == ".txt":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UploadValidationError("TXT uploads must contain valid UTF-8.") from exc
        if not text.strip():
            raise UploadValidationError("TXT uploads must contain non-whitespace text.")
        return
    if not content.startswith(b"%PDF-"):
        raise UploadValidationError("PDF uploads have an invalid signature.")
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise UploadValidationError("PDF uploads could not be parsed.") from exc
    if not text.strip():
        raise UploadValidationError("PDF uploads must contain extractable text.")


def _prepare_uploads(
    context: _VectorContext,
    uploads: Sequence[_UploadedFile],
) -> list[_PreparedUpload]:
    if not 1 <= len(uploads) <= _MAX_UPLOAD_FILES:
        raise UploadLimitExceededError("Upload batches must contain between 1 and 10 files.")
    if any(len(upload.content) > _MAX_FILE_BYTES for upload in uploads):
        raise UploadLimitExceededError("An uploaded file exceeds the 25 MiB limit.")
    if sum(len(upload.content) for upload in uploads) > _MAX_TOTAL_BYTES:
        raise UploadLimitExceededError("The upload batch exceeds the 100 MiB limit.")

    prepared: list[_PreparedUpload] = []
    document_ids: set[str] = set()
    for upload in uploads:
        filename = _safe_basename(upload.filename)
        _validate_upload_content(filename, upload.content)
        content_hash = hashlib.sha256(upload.content).hexdigest()
        upload_key = hashlib.sha256(f"{filename}\0{content_hash}".encode()).hexdigest()
        relative_path = (Path(upload_key) / filename).as_posix()
        target = context.settings.sources_dir / relative_path
        sources_root = context.settings.sources_dir.resolve()
        child = context.settings.sources_dir
        for part in Path(relative_path).parts:
            child /= part
            if child.is_symlink():
                raise UploadValidationError("The upload target path is invalid.")
        if not target.resolve().is_relative_to(sources_root):
            raise UploadValidationError("The upload target path is invalid.")
        document_id = context.document_id(target)
        if document_id in document_ids:
            raise UploadValidationError("The upload batch contains duplicate documents.")
        document_ids.add(document_id)
        prepared.append(
            _PreparedUpload(
                filename=filename,
                content=upload.content,
                target=target,
                relative_path=relative_path,
                document_id=document_id,
            )
        )
    return prepared


def _restore_manifest(context: _VectorContext, previous_bytes: bytes | None) -> None:
    manifest_path = context.settings.manifest_path
    temporary = manifest_path.with_suffix(".tmp")
    if temporary.exists():
        temporary.unlink()
    if previous_bytes is None:
        if manifest_path.exists():
            manifest_path.unlink()
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(previous_bytes)
    os.replace(temporary, manifest_path)


def _updated_manifest(context: _VectorContext, *, request: _ManifestUpdate) -> IngestionManifest:
    document_id = context.document_id(request.source)
    updated = request.previous.model_copy(deep=True)
    previous_record = updated.documents.get(document_id)
    now = datetime.now(timezone.utc)
    if (
        previous_record is not None
        and previous_record.updated_at is not None
        and now <= previous_record.updated_at
    ):
        now = previous_record.updated_at + timedelta(microseconds=1)
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
        size_bytes=request.source.stat().st_size,
        indexed_at=(
            previous_record.indexed_at
            if previous_record is not None and previous_record.indexed_at is not None
            else now
        ),
        updated_at=now,
    )
    return updated


@dataclass(frozen=True)
class _StoredVectorSnapshot:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    embeddings: list[list[float]]


def _stored_vectors(store: Chroma, *, chunk_ids: set[str]) -> _StoredVectorSnapshot:
    if not chunk_ids:
        return _StoredVectorSnapshot([], [], [], [])
    payload = store._collection.get(
        ids=sorted(chunk_ids),
        include=cast(Any, ["documents", "metadatas", "embeddings"]),
    )
    raw_embeddings = payload.get("embeddings")
    return _StoredVectorSnapshot(
        ids=[str(item) for item in payload["ids"]],
        documents=[str(item) for item in payload.get("documents") or []],
        metadatas=[
            cast(dict[str, Any], dict(item or {})) for item in payload.get("metadatas") or []
        ],
        embeddings=[[float(value) for value in embedding] for embedding in raw_embeddings]
        if raw_embeddings is not None
        else [],
    )


def _best_effort_restore_vectors(
    store: Chroma | None,
    *,
    touched_ids: set[str],
    previous: _StoredVectorSnapshot,
) -> None:
    if store is None:
        return
    try:
        if touched_ids:
            store._collection.delete(ids=sorted(touched_ids))
    except Exception:
        pass
    try:
        if previous.ids:
            store._collection.upsert(
                ids=previous.ids,
                documents=previous.documents,
                metadatas=cast(Any, previous.metadatas),
                embeddings=cast(Any, previous.embeddings),
            )
    except Exception:
        pass


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

    def index_upload_batch(
        self: _VectorContext,
        uploads: Sequence[_UploadedFile],
    ) -> list[ManifestDocument]:
        records, _manifest = self.index_upload_batch_with_manifest(uploads)
        return records

    def index_upload_batch_with_manifest(
        self: _VectorContext,
        uploads: Sequence[_UploadedFile],
    ) -> tuple[list[ManifestDocument], IngestionManifest]:
        prepared = _prepare_uploads(self, uploads)
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        previous_manifest = self.manifest()
        manifest_path = self.settings.manifest_path
        previous_manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
        old_ids: set[str] = set()
        for upload in prepared:
            previous_record = previous_manifest.documents.get(upload.document_id)
            if previous_record is not None:
                old_ids.update(previous_record.chunk_ids)
        touched_ids: set[str] = set()
        store: Chroma | None = None
        previous_vectors = _StoredVectorSnapshot([], [], [], [])
        staging = Path(tempfile.mkdtemp(prefix="rag-upload-", dir=self.settings.data_dir))
        committed: tuple[list[ManifestDocument], IngestionManifest] | None = None
        try:
            staged: list[tuple[_PreparedUpload, Path, Path | None]] = []
            for index, upload in enumerate(prepared):
                staged_source = staging / "incoming" / str(index) / upload.filename
                staged_source.parent.mkdir(parents=True, exist_ok=True)
                staged_source.write_bytes(upload.content)
                backup = None
                if upload.target.exists():
                    backup = staging / "backups" / str(index) / upload.filename
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(upload.target, backup)
                staged.append((upload, staged_source, backup))

            try:
                for upload, staged_source, _backup in staged:
                    upload.target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staged_source, upload.target)

                store = self._store()
                previous_vectors = _stored_vectors(store, chunk_ids=old_ids)
                updated_manifest = previous_manifest.model_copy(deep=True)
                records: list[ManifestDocument] = []
                for upload, _staged_source, _backup in staged:
                    chunks = self.prepare_chunks(self.load_documents([upload.target]))
                    if not chunks:
                        raise VectorTransactionError(
                            "Uploaded content produced no indexable chunks."
                        )
                    chunk_ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
                    previous_record = updated_manifest.documents.get(upload.document_id)
                    previous_ids = set(previous_record.chunk_ids) if previous_record else set()
                    touched_ids.update((*previous_ids, *chunk_ids))
                    store.add_documents(chunks, ids=chunk_ids)
                    if set(store.get(ids=chunk_ids)["ids"]) != set(chunk_ids):
                        raise VectorTransactionError("Uploaded chunks could not be verified.")
                    stale_ids = previous_ids - set(chunk_ids)
                    if stale_ids:
                        store.delete(ids=sorted(stale_ids))
                    updated_manifest = _updated_manifest(
                        self,
                        request=_ManifestUpdate(
                            previous=updated_manifest,
                            source=upload.target,
                            chunks=chunks,
                            chunk_ids=chunk_ids,
                        ),
                    )
                    records.append(updated_manifest.documents[upload.document_id])
                self._write_manifest(updated_manifest)
                committed = records, updated_manifest.model_copy(deep=True)
            except Exception as exc:
                _best_effort_restore_vectors(
                    store,
                    touched_ids=touched_ids,
                    previous=previous_vectors,
                )
                for upload, _staged_source, backup in reversed(staged):
                    try:
                        if backup is not None:
                            upload.target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(backup, upload.target)
                        elif upload.target.exists():
                            upload.target.unlink()
                        if upload.target.parent != self.settings.sources_dir:
                            try:
                                upload.target.parent.rmdir()
                            except OSError:
                                pass
                    except Exception:
                        pass
                try:
                    _restore_manifest(self, previous_manifest_bytes)
                except Exception:
                    pass
                if isinstance(exc, (UploadValidationError, UploadLimitExceededError)):
                    raise
                raise VectorTransactionError("The upload batch could not be indexed.") from exc
        finally:
            try:
                shutil.rmtree(staging)
            except Exception:
                pass
        if committed is None:
            raise VectorTransactionError("The upload batch could not be indexed.")
        return committed

    def index_document(self: _VectorContext, path: str | Path) -> IngestionResult:
        source = Path(path)
        document_id = self.document_id(source)
        manifest = self.manifest()
        previous_record = manifest.documents.get(document_id)
        old_ids = set(previous_record.chunk_ids) if previous_record else set()
        touched_ids: set[str] = set()
        store: Chroma | None = None
        previous_vectors = _StoredVectorSnapshot([], [], [], [])
        try:
            chunks = self.prepare_chunks(self.load_documents([source]))
            if not chunks:
                raise ValueError("No supported content was parsed")
            ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            store = self._store()
            previous_vectors = _stored_vectors(store, chunk_ids=old_ids)
            touched_ids.update((*old_ids, *ids))
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
            _best_effort_restore_vectors(
                store,
                touched_ids=touched_ids,
                previous=previous_vectors,
            )
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
        return (
            self.delete_document_with_manifest(
                document_id,
                delete_source=delete_source,
            )
            is not None
        )

    def delete_document_with_manifest(
        self: _VectorContext,
        document_id: str,
        *,
        delete_source: bool = True,
    ) -> IngestionManifest | None:
        previous_manifest = self.manifest()
        record = previous_manifest.documents.get(document_id)
        if record is None:
            return None
        source = (self.settings.sources_dir / record.relative_path).resolve()
        sources_root = self.settings.sources_dir.resolve()
        if not source.is_relative_to(sources_root):
            raise VectorTransactionError("The document source path is invalid.")

        manifest_path = self.settings.manifest_path
        previous_manifest_bytes = manifest_path.read_bytes() if manifest_path.exists() else None
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        store: Chroma | None = None
        touched_ids: set[str] = set()
        previous_vectors = _StoredVectorSnapshot([], [], [], [])
        staging = Path(tempfile.mkdtemp(prefix="rag-delete-", dir=self.settings.data_dir))
        committed: IngestionManifest | None = None
        try:
            backup = staging / source.name
            had_source = delete_source and source.exists()
            if had_source:
                shutil.copy2(source, backup)
            try:
                store = self._store()
                previous_vectors = _stored_vectors(store, chunk_ids=set(record.chunk_ids))
                if record.chunk_ids:
                    touched_ids.update(record.chunk_ids)
                    store.delete(ids=record.chunk_ids)
                updated = previous_manifest.model_copy(deep=True)
                del updated.documents[document_id]
                self._write_manifest(updated)
                if had_source:
                    source.unlink()
                    if source.parent != sources_root:
                        try:
                            source.parent.rmdir()
                        except OSError:
                            pass
                committed = updated.model_copy(deep=True)
            except Exception as exc:
                _best_effort_restore_vectors(
                    store,
                    touched_ids=touched_ids,
                    previous=previous_vectors,
                )
                try:
                    _restore_manifest(self, previous_manifest_bytes)
                except Exception:
                    pass
                try:
                    if had_source:
                        source.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup, source)
                except Exception:
                    pass
                raise VectorTransactionError("The document could not be deleted.") from exc
        finally:
            try:
                shutil.rmtree(staging)
            except Exception:
                pass
        if committed is None:
            raise VectorTransactionError("The document could not be deleted.")
        return committed

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
