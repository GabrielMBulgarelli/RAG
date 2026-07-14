import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from modules.config import Settings
from modules.vector_db import VectorDBManager


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        total = sum(text.encode("utf-8"))
        return [float(total % 97), float(len(text)), 1.0]


@pytest.fixture
def manager(tmp_path: Path) -> VectorDBManager:
    settings = Settings(
        sources_dir=tmp_path / "sources",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        manifest_path=tmp_path / "data" / "manifest.json",
        trace_dir=tmp_path / "data" / "traces",
        logs_dir=tmp_path / "logs",
        chunk_size=40,
        chunk_overlap=5,
    )
    return VectorDBManager(settings=settings, embeddings=FakeEmbeddings())


def write_source(manager: VectorDBManager, relative: str, content: str) -> Path:
    path = manager.settings.sources_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_unchanged_reindex_does_not_increase_chunk_count(manager: VectorDBManager) -> None:
    path = write_source(manager, "guide.txt", "alpha beta gamma " * 8)

    manager.index_documents([path])
    first_count = manager.chunk_count()
    manager.index_documents([path])

    assert manager.chunk_count() == first_count


def test_modified_document_removes_stale_chunks(manager: VectorDBManager) -> None:
    path = write_source(manager, "guide.txt", "old material " * 10)
    manager.index_documents([path])
    old_ids = set(manager.manifest().documents.values().__iter__().__next__().chunk_ids)

    path.write_text("replacement material " * 4, encoding="utf-8")
    manager.index_documents([path])
    record = next(iter(manager.manifest().documents.values()))
    stored_ids = set(manager._store().get()["ids"])

    assert not (old_ids - set(record.chunk_ids)) & stored_ids
    assert stored_ids == set(record.chunk_ids)


def test_delete_document_removes_chunks_and_manifest_record(manager: VectorDBManager) -> None:
    path = write_source(manager, "guide.txt", "delete this document")
    manager.index_documents([path])
    document_id = next(iter(manager.manifest().documents))

    manager.delete_document(document_id)

    assert manager.chunk_count() == 0
    assert document_id not in manager.manifest().documents


def test_same_basename_in_different_paths_remains_distinct(manager: VectorDBManager) -> None:
    first = write_source(manager, "one/report.txt", "first report")
    second = write_source(manager, "two/report.txt", "second report")

    manager.index_documents([first, second])
    records = list(manager.manifest().documents.values())

    assert len(records) == 2
    assert len({record.document_id for record in records}) == 2
    assert {record.relative_path for record in records} == {"one/report.txt", "two/report.txt"}


def test_page_metadata_survives_persistence_reload(manager: VectorDBManager) -> None:
    path = write_source(manager, "guide.txt", "persistent page metadata")
    manager.index_documents([path])
    manager.vectorstore = None

    metadata = manager._store().get(include=["metadatas"])["metadatas"][0]

    assert metadata["filename"] == "guide.txt"
    assert metadata["page"] == 1
    assert metadata["document_id"]


def test_failed_parsing_preserves_previous_version(
    manager: VectorDBManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_source(manager, "guide.txt", "usable version")
    manager.index_documents([path])
    before_manifest = manager.manifest().model_dump()
    before_ids = set(manager._store().get()["ids"])

    def fail(_paths=None):
        raise ValueError("parse failed")

    monkeypatch.setattr(manager, "load_documents", fail)
    result = manager.index_document(path)

    assert not result.success
    assert result.error is not None
    assert manager.manifest().model_dump() == before_manifest
    assert set(manager._store().get()["ids"]) == before_ids


def test_manifest_is_always_valid_json_after_replacement(manager: VectorDBManager) -> None:
    path = write_source(manager, "guide.txt", "first version")
    manager.index_documents([path])
    path.write_text("second version", encoding="utf-8")
    manager.index_documents([path])

    payload = json.loads(manager.settings.manifest_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert not manager.settings.manifest_path.with_suffix(".tmp").exists()


def test_reconciliation_detects_missing_and_orphaned_chunks(manager: VectorDBManager) -> None:
    path = write_source(manager, "guide.txt", "known document")
    manager.index_documents([path])
    known_id = next(iter(manager._store().get()["ids"]))
    manager._store().delete(ids=[known_id])
    manager._store().add_documents(
        [Document(page_content="orphan", metadata={"chunk_id": "orphan"})], ids=["orphan"]
    )

    result = manager.reconcile_index()

    assert known_id in result.missing_chunk_ids
    assert "orphan" in result.orphan_chunk_ids
