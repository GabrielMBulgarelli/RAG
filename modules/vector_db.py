"""Persistent PDF/TXT ingestion with stable Chroma chunk IDs."""

import hashlib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, config
from .models import (
    IngestionError,
    IngestionManifest,
    IngestionResult,
    ManifestDocument,
    ReconciliationResult,
)


class VectorDBManager:
    def __init__(self, settings: Settings | None = None, embeddings: Embeddings | None = None):
        self.settings = settings or config
        self.embeddings = embeddings or OllamaEmbeddings(
            model=self.settings.embedding_model, base_url=self.settings.ollama_base_url
        )
        self.vectorstore: Chroma | None = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.settings.sources_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    def document_id(self, path: str | Path) -> str:
        relative_path = self._relative_path(Path(path))
        return hashlib.sha256(relative_path.encode()).hexdigest()

    def manifest(self) -> IngestionManifest:
        path = self.settings.manifest_path
        if not path.exists():
            return IngestionManifest()
        return IngestionManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: IngestionManifest) -> None:
        path = self.settings.manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        payload = manifest.model_dump_json(indent=2)
        temporary.write_text(payload, encoding="utf-8")
        IngestionManifest.model_validate_json(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)

    def load_documents(self, paths: Sequence[str | Path] | None = None) -> list[Document]:
        if paths is None:
            self.settings.sources_dir.mkdir(parents=True, exist_ok=True)
            paths = sorted(
                (
                    *self.settings.sources_dir.rglob("*.pdf"),
                    *self.settings.sources_dir.rglob("*.txt"),
                )
            )
        documents: list[Document] = []
        for raw_path in paths:
            path = Path(raw_path)
            if path.suffix.lower() == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif path.suffix.lower() == ".txt":
                loaded = TextLoader(str(path), encoding="utf-8").load()
            else:
                continue
            relative_path = self._relative_path(path)
            document_id = self.document_id(path)
            for document in loaded:
                page = (
                    int(document.metadata.get("page", 0)) + 1
                    if path.suffix.lower() == ".pdf"
                    else 1
                )
                document.metadata = {
                    "document_id": document_id,
                    "relative_path": relative_path,
                    "filename": path.name,
                    "page": page,
                }
            documents.extend(loaded)
        return documents

    def prepare_chunks(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            for chunk_index, chunk in enumerate(self.text_splitter.split_documents([document])):
                filename = str(chunk.metadata.get("filename", "Unknown"))
                page = int(chunk.metadata.get("page", 1))
                document_id = str(
                    chunk.metadata.get("document_id")
                    or hashlib.sha256(filename.encode()).hexdigest()
                )
                content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
                digest = hashlib.sha256(
                    f"{document_id}:{page}:{chunk_index}:{content_hash}".encode()
                ).hexdigest()
                chunk.metadata.update(
                    {
                        "chunk_id": digest,
                        "document_id": document_id,
                        "filename": filename,
                        "page": page,
                        "chunk": chunk_index,
                    }
                )
                chunks.append(chunk)
        return chunks

    def _store(self) -> Chroma:
        if self.vectorstore is None:
            self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name="rag_documents",
                persist_directory=str(self.settings.chroma_dir),
                embedding_function=self.embeddings,
            )
        return self.vectorstore

    def index_documents(self, paths: Sequence[str | Path] | None = None) -> int:
        if paths is None:
            self.settings.sources_dir.mkdir(parents=True, exist_ok=True)
            paths = sorted(
                (
                    *self.settings.sources_dir.rglob("*.pdf"),
                    *self.settings.sources_dir.rglob("*.txt"),
                )
            )
        return sum(
            result.chunk_count
            for result in (self.index_document(Path(path)) for path in paths)
            if result.success
        )

    def index_document(self, path: str | Path) -> IngestionResult:
        source = Path(path)
        document_id = self.document_id(source)
        previous_manifest = self.manifest()
        previous = previous_manifest.documents.get(document_id)
        inserted_ids: set[str] = set()
        previous_chunks: list[Document] = []
        try:
            chunks = self.prepare_chunks(self.load_documents([source]))
            if not chunks:
                raise ValueError("No supported content was parsed")
            ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            old_ids = set(previous.chunk_ids if previous else [])
            store = self._store()
            if old_ids:
                old_payload = store.get(ids=sorted(old_ids), include=["documents", "metadatas"])
                previous_chunks = [
                    Document(page_content=content, metadata=metadata or {})
                    for content, metadata in zip(
                        old_payload.get("documents") or [],
                        old_payload.get("metadatas") or [],
                        strict=True,
                    )
                ]
            inserted_ids = set(ids) - old_ids
            store.add_documents(chunks, ids=ids)
            persisted = set(store.get(ids=ids)["ids"])
            if persisted != set(ids):
                raise RuntimeError("New chunks could not be verified after upsert")

            stale_ids = old_ids - set(ids)
            if stale_ids:
                store.delete(ids=sorted(stale_ids))
            content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            pages = {int(chunk.metadata.get("page", 1)) for chunk in chunks}
            updated = previous_manifest.model_copy(deep=True)
            updated.documents[document_id] = ManifestDocument(
                document_id=document_id,
                relative_path=self._relative_path(source),
                filename=source.name,
                content_hash=content_hash,
                chunk_ids=ids,
                page_count=len(pages),
                chunk_count=len(ids),
                embedding_model=self.settings.embedding_model,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self._write_manifest(updated)
            return IngestionResult(document_id=document_id, success=True, chunk_count=len(ids))
        except Exception as exc:
            if inserted_ids and self.vectorstore is not None:
                self.vectorstore.delete(ids=sorted(inserted_ids))
            if previous_chunks and self.vectorstore is not None:
                self.vectorstore.add_documents(
                    previous_chunks,
                    ids=[str(chunk.metadata["chunk_id"]) for chunk in previous_chunks],
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

    def save_uploads(self, paths: list[str] | None) -> list[Path]:
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

    def rebuild(self) -> int:
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
        elif self.settings.chroma_dir.exists():
            shutil.rmtree(self.settings.chroma_dir)
        if self.settings.manifest_path.exists():
            self.settings.manifest_path.unlink()
        return self.index_documents()

    def delete_document(self, document_id: str, *, delete_source: bool = True) -> bool:
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

    def reconcile_index(self) -> ReconciliationResult:
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

    def setup(self, force_rebuild: bool = False) -> Chroma:
        if force_rebuild:
            self.rebuild()
        return self._store()

    def get_retriever(self, search_kwargs: dict | None = None):
        return self._store().as_retriever(
            search_kwargs=search_kwargs or {"k": self.settings.k_retrieval}
        )

    def document_names(self) -> list[str]:
        stored = self._store().get(include=["metadatas"])
        return sorted(
            {str(item.get("filename", "Unknown")) for item in stored.get("metadatas") or []}
        )

    def chunk_count(self) -> int:
        return int(self._store()._collection.count())
