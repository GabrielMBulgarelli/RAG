"""Persistent PDF/TXT ingestion with stable Chroma chunk IDs."""

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, config
from .models import IngestionManifest
from .vector_db_operations import VectorIngestionMixin, VectorMaintenanceMixin


class VectorDBManager(VectorIngestionMixin, VectorMaintenanceMixin):
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
