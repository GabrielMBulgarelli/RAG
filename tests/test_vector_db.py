import hashlib
import json
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

import modules.vector_db_operations as vector_operations
from modules.application.models import UploadedFile
from modules.config import Settings
from modules.vector_db import VectorDBManager
from modules.vector_db_operations import (
    UploadLimitExceededError,
    UploadValidationError,
    VectorTransactionError,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        total = sum(text.encode("utf-8"))
        return [float(total % 97), float(len(text)), 1.0]


class ArmedEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        self._calls = 0
        self._fail_on: int | None = None

    def arm(self, *, fail_on: int) -> None:
        self._calls = 0
        self._fail_on = fail_on

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._calls += 1
        if self._fail_on is not None and self._calls >= self._fail_on:
            raise RuntimeError("embedding unavailable")
        return super().embed_documents(texts)


def raw_vector_snapshot(manager: VectorDBManager) -> dict[str, tuple[str, dict, list[float]]]:
    payload = manager._store()._collection.get(
        include=cast(Any, ["documents", "metadatas", "embeddings"])
    )
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    embeddings = payload.get("embeddings")
    assert embeddings is not None
    return {
        str(chunk_id): (
            str(documents[index]),
            dict(metadatas[index] or {}),
            [float(value) for value in embeddings[index]],
        )
        for index, chunk_id in enumerate(payload["ids"])
    }


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


def blank_pdf() -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(output)
    return output.getvalue()


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
    assert not path.exists()
    reloaded = VectorDBManager(settings=manager.settings, embeddings=FakeEmbeddings())
    assert document_id not in reloaded.manifest().documents


def test_failed_manifest_step_restores_delete_transaction(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_source(manager, "guide.txt", "keep this document")
    assert manager.index_document(path).success
    document_id = next(iter(manager.manifest().documents))
    before_manifest = manager.settings.manifest_path.read_bytes()
    before_ids = set(manager._store().get()["ids"])
    before_source = path.read_bytes()

    def fail_write(_manifest) -> None:
        raise OSError("manifest failure")

    monkeypatch.setattr(manager, "_write_manifest", fail_write)

    with pytest.raises(VectorTransactionError):
        manager.delete_document(document_id)

    assert manager.settings.manifest_path.read_bytes() == before_manifest
    assert set(manager._store().get()["ids"]) == before_ids
    assert path.read_bytes() == before_source


def test_failed_source_step_restores_delete_transaction(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_source(manager, "guide.txt", "keep source bytes")
    assert manager.index_document(path).success
    document_id = next(iter(manager.manifest().documents))
    before_manifest = manager.settings.manifest_path.read_bytes()
    before_ids = set(manager._store().get()["ids"])
    real_unlink = Path.unlink

    def fail_source_unlink(target: Path, *args, **kwargs) -> None:
        if target.resolve() == path.resolve():
            raise OSError("source failure")
        real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_unlink)

    with pytest.raises(VectorTransactionError):
        manager.delete_document(document_id)

    assert manager.settings.manifest_path.read_bytes() == before_manifest
    assert set(manager._store().get()["ids"]) == before_ids
    assert path.read_bytes() == b"keep source bytes"


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

    assert payload["schema_version"] == 2
    record = next(iter(payload["documents"].values()))
    assert record["size_bytes"] == len("second version".encode())
    assert record["indexed_at"] is not None
    assert record["updated_at"] is not None
    assert not manager.settings.manifest_path.with_suffix(".tmp").exists()


def test_schema_v1_manifest_loads_unchanged_and_reindex_sets_missing_timestamp(
    manager: VectorDBManager,
) -> None:
    path = write_source(manager, "legacy/guide.txt", "legacy document")
    document_id = manager.document_id(path)
    manager.settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manager.settings.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": {
                    document_id: {
                        "document_id": document_id,
                        "relative_path": "legacy/guide.txt",
                        "filename": "guide.txt",
                        "content_hash": "legacy-hash",
                        "chunk_ids": ["legacy-chunk"],
                        "page_count": 1,
                        "chunk_count": 1,
                        "embedding_model": manager.settings.embedding_model,
                        "chunk_size": manager.settings.chunk_size,
                        "chunk_overlap": manager.settings.chunk_overlap,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = manager.manifest()
    assert loaded.documents[document_id].document_id == document_id
    assert loaded.documents[document_id].relative_path == "legacy/guide.txt"
    assert loaded.documents[document_id].chunk_ids == ["legacy-chunk"]
    manager._write_manifest(loaded)
    assert (
        json.loads(manager.settings.manifest_path.read_text(encoding="utf-8"))["schema_version"]
        == 2
    )

    result = manager.index_document(path)

    assert result.success
    reindexed = manager.manifest().documents[document_id]
    assert reindexed.indexed_at is not None
    assert reindexed.updated_at is not None


def test_atomic_upload_batch_commits_all_records_and_safe_sources(
    manager: VectorDBManager,
) -> None:
    records = manager.index_upload_batch(
        (
            UploadedFile(
                filename="../../guide.txt",
                content_type="text/plain",
                content=b"alpha guide",
            ),
            UploadedFile(
                filename="notes.TXT",
                content_type="text/plain",
                content=b"beta notes",
            ),
        )
    )

    manifest = manager.manifest()
    assert {record.document_id for record in records} == set(manifest.documents)
    assert {record.filename for record in records} == {"guide.txt", "notes.TXT"}
    assert all(
        (manager.settings.sources_dir / record.relative_path).is_file() for record in records
    )
    assert all(
        (manager.settings.sources_dir / record.relative_path)
        .resolve()
        .is_relative_to(manager.settings.sources_dir.resolve())
        for record in records
    )
    assert set(manager._store().get()["ids"]) == {
        chunk_id for record in records for chunk_id in record.chunk_ids
    }
    assert not list(manager.settings.data_dir.glob("rag-upload-*"))


def test_upload_rejects_existing_symlink_component_before_staging(
    manager: VectorDBManager,
    tmp_path: Path,
) -> None:
    upload = UploadedFile(
        filename="guide.txt",
        content_type="text/plain",
        content=b"safe content",
    )
    content_hash = hashlib.sha256(upload.content).hexdigest()
    upload_key = hashlib.sha256(f"{upload.filename}\0{content_hash}".encode()).hexdigest()
    manager.settings.sources_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = manager.settings.sources_dir / upload_key
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(UploadValidationError):
        manager.index_upload_batch((upload,))

    assert not (outside / upload.filename).exists()
    assert not manager.settings.data_dir.exists()
    assert not manager.settings.manifest_path.exists()


@pytest.mark.parametrize(
    "uploads",
    [
        (),
        tuple(
            UploadedFile(filename=f"{index}.txt", content_type="text/plain", content=b"x")
            for index in range(11)
        ),
        (
            UploadedFile(
                filename="large.txt",
                content_type="text/plain",
                content=b"x" * (25 * 1024 * 1024 + 1),
            ),
        ),
        tuple(
            UploadedFile(
                filename=f"{index}.txt",
                content_type="text/plain",
                content=b"x" * (10 * 1024 * 1024 + 1),
            )
            for index in range(10)
        ),
    ],
)
def test_upload_batch_enforces_count_and_byte_limits(
    manager: VectorDBManager,
    uploads: tuple[UploadedFile, ...],
) -> None:
    with pytest.raises(UploadLimitExceededError):
        manager.index_upload_batch(uploads)

    assert not manager.settings.sources_dir.exists()
    assert not manager.settings.manifest_path.exists()


@pytest.mark.parametrize(
    "upload",
    [
        UploadedFile(filename="guide.csv", content_type="text/csv", content=b"value"),
        UploadedFile(filename="guide.txt", content_type="text/plain", content=b"\xff"),
        UploadedFile(filename="guide.txt", content_type="text/plain", content=b" \n\t "),
        UploadedFile(filename="guide.pdf", content_type="application/pdf", content=b"not pdf"),
        UploadedFile(filename="guide.pdf", content_type="application/pdf", content=b"%PDF-broken"),
        UploadedFile(filename="guide.pdf", content_type="application/pdf", content=blank_pdf()),
    ],
)
def test_upload_batch_rejects_invalid_or_empty_content(
    manager: VectorDBManager,
    upload: UploadedFile,
) -> None:
    with pytest.raises(UploadValidationError):
        manager.index_upload_batch((upload,))

    assert not manager.settings.sources_dir.exists()
    assert not manager.settings.manifest_path.exists()


def test_upload_batch_rejects_duplicate_targets_before_mutation(
    manager: VectorDBManager,
) -> None:
    upload = UploadedFile(filename="guide.txt", content_type="text/plain", content=b"same")

    with pytest.raises(UploadValidationError, match="duplicate"):
        manager.index_upload_batch((upload, upload))

    assert not manager.settings.sources_dir.exists()
    assert not manager.settings.manifest_path.exists()


def test_exact_reupload_preserves_document_id_and_does_not_duplicate_chunks(
    manager: VectorDBManager,
) -> None:
    upload = UploadedFile(
        filename="guide.txt",
        content_type="text/plain",
        content=b"same document content",
    )

    first = manager.index_upload_batch((upload,))[0]
    first_count = manager.chunk_count()
    second = manager.index_upload_batch((upload,))[0]

    assert second.document_id == first.document_id
    assert second.indexed_at == first.indexed_at
    assert second.updated_at is not None
    assert first.updated_at is not None
    assert second.updated_at >= first.updated_at
    assert manager.chunk_count() == first_count
    assert not list(manager.settings.data_dir.glob("rag-upload-*"))


def test_reupload_updated_at_is_strictly_monotonic_when_clock_does_not_advance(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upload = UploadedFile(
        filename="guide.txt",
        content_type="text/plain",
        content=b"same document content",
    )
    first = manager.index_upload_batch((upload,))[0]
    assert first.updated_at is not None

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return first.updated_at

    monkeypatch.setattr(vector_operations, "datetime", FrozenDateTime)

    second = manager.index_upload_batch((upload,))[0]

    assert second.updated_at is not None
    assert second.updated_at > first.updated_at


def test_second_file_embedding_failure_restores_exact_vectors_without_reembedding(
    manager: VectorDBManager,
) -> None:
    embeddings = ArmedEmbeddings()
    vector_db = VectorDBManager(settings=manager.settings, embeddings=embeddings)
    existing = UploadedFile(
        filename="guide.txt",
        content_type="text/plain",
        content=b"existing guide",
    )
    unrelated = UploadedFile(
        filename="unrelated.txt",
        content_type="text/plain",
        content=b"unrelated document",
    )
    vector_db.index_upload_batch((existing, unrelated))
    before_vectors = raw_vector_snapshot(vector_db)
    before_manifest = vector_db.settings.manifest_path.read_bytes()
    before_sources = {
        path.relative_to(vector_db.settings.sources_dir): path.read_bytes()
        for path in vector_db.settings.sources_dir.rglob("*.txt")
    }
    embeddings.arm(fail_on=2)

    with pytest.raises(VectorTransactionError):
        vector_db.index_upload_batch(
            (
                existing,
                UploadedFile(
                    filename="new.txt",
                    content_type="text/plain",
                    content=b"new document",
                ),
            )
        )

    assert raw_vector_snapshot(vector_db) == before_vectors
    assert vector_db.settings.manifest_path.read_bytes() == before_manifest
    assert {
        path.relative_to(vector_db.settings.sources_dir): path.read_bytes()
        for path in vector_db.settings.sources_dir.rglob("*.txt")
    } == before_sources


def test_upload_snapshot_failure_does_not_delete_existing_vectors(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = UploadedFile(
        filename="guide.txt",
        content_type="text/plain",
        content=b"existing guide",
    )
    unrelated = UploadedFile(
        filename="unrelated.txt",
        content_type="text/plain",
        content=b"unrelated document",
    )
    manager.index_upload_batch((existing, unrelated))
    before_vectors = raw_vector_snapshot(manager)
    before_manifest = manager.settings.manifest_path.read_bytes()
    before_sources = {
        path.relative_to(manager.settings.sources_dir): path.read_bytes()
        for path in manager.settings.sources_dir.rglob("*.txt")
    }
    collection = manager._store()._collection
    real_get = collection.get

    def fail_snapshot(*args, **kwargs):
        raise OSError("snapshot failure")

    monkeypatch.setattr(collection, "get", fail_snapshot)

    with pytest.raises(VectorTransactionError):
        manager.index_upload_batch((existing,))

    monkeypatch.setattr(collection, "get", real_get)
    assert raw_vector_snapshot(manager) == before_vectors
    assert manager.settings.manifest_path.read_bytes() == before_manifest
    assert {
        path.relative_to(manager.settings.sources_dir): path.read_bytes()
        for path in manager.settings.sources_dir.rglob("*.txt")
    } == before_sources


def test_single_reindex_snapshot_failure_does_not_delete_existing_vectors(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = write_source(manager, "target.txt", "target document")
    unrelated = write_source(manager, "unrelated.txt", "unrelated document")
    assert manager.index_document(target).success
    assert manager.index_document(unrelated).success
    before_vectors = raw_vector_snapshot(manager)
    before_manifest = manager.settings.manifest_path.read_bytes()
    before_sources = {
        path.relative_to(manager.settings.sources_dir): path.read_bytes()
        for path in manager.settings.sources_dir.rglob("*.txt")
    }
    collection = manager._store()._collection
    real_get = collection.get

    def fail_snapshot(*args, **kwargs):
        raise OSError("snapshot failure")

    monkeypatch.setattr(collection, "get", fail_snapshot)

    result = manager.index_document(target)

    monkeypatch.setattr(collection, "get", real_get)
    assert result.success is False
    assert raw_vector_snapshot(manager) == before_vectors
    assert manager.settings.manifest_path.read_bytes() == before_manifest
    assert {
        path.relative_to(manager.settings.sources_dir): path.read_bytes()
        for path in manager.settings.sources_dir.rglob("*.txt")
    } == before_sources


def test_second_upload_failure_restores_manifest_chunks_and_sources(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = manager.load_documents
    calls = 0

    def fail_second(paths=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("parse failure")
        return real_load(paths)

    monkeypatch.setattr(manager, "load_documents", fail_second)

    with pytest.raises(VectorTransactionError):
        manager.index_upload_batch(
            (
                UploadedFile(filename="one.txt", content_type="text/plain", content=b"one"),
                UploadedFile(filename="two.txt", content_type="text/plain", content=b"two"),
            )
        )

    assert not manager.settings.manifest_path.exists()
    assert set(manager._store().get()["ids"]) == set()
    assert not list(manager.settings.sources_dir.rglob("*"))
    assert not list(manager.settings.data_dir.glob("rag-upload-*"))


def test_upload_cleanup_failure_after_commit_does_not_change_success(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_rmtree = shutil.rmtree

    def cleanup_then_fail(path, *args, **kwargs) -> None:
        real_rmtree(path, *args, **kwargs)
        raise OSError("cleanup failure")

    monkeypatch.setattr(vector_operations.shutil, "rmtree", cleanup_then_fail)

    records = manager.index_upload_batch(
        (
            UploadedFile(
                filename="guide.txt",
                content_type="text/plain",
                content=b"committed guide",
            ),
        )
    )

    assert len(records) == 1
    assert records[0].document_id in manager.manifest().documents
    assert set(manager._store().get()["ids"]) == set(records[0].chunk_ids)


def test_manifest_write_failure_restores_unrelated_state(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = write_source(manager, "existing.txt", "existing document")
    assert manager.index_document(existing).success
    before_manifest = manager.settings.manifest_path.read_bytes()
    before_ids = set(manager._store().get()["ids"])
    before_source = existing.read_bytes()

    def fail_write(_manifest) -> None:
        raise OSError("manifest failure")

    monkeypatch.setattr(manager, "_write_manifest", fail_write)

    with pytest.raises(VectorTransactionError):
        manager.index_upload_batch(
            (
                UploadedFile(
                    filename="new.txt",
                    content_type="text/plain",
                    content=b"new document",
                ),
            )
        )

    assert manager.settings.manifest_path.read_bytes() == before_manifest
    assert set(manager._store().get()["ids"]) == before_ids
    assert existing.read_bytes() == before_source
    assert not list(manager.settings.data_dir.glob("rag-upload-*"))


def test_delete_rollback_restores_exact_vectors_without_embedding(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = ArmedEmbeddings()
    vector_db = VectorDBManager(settings=manager.settings, embeddings=embeddings)
    target = write_source(vector_db, "target.txt", "target document")
    unrelated = write_source(vector_db, "unrelated.txt", "unrelated document")
    assert vector_db.index_document(target).success
    assert vector_db.index_document(unrelated).success
    target_id = vector_db.document_id(target)
    before_vectors = raw_vector_snapshot(vector_db)
    before_manifest = vector_db.settings.manifest_path.read_bytes()
    embeddings.arm(fail_on=1)

    def fail_write(_manifest) -> None:
        raise OSError("manifest failure")

    monkeypatch.setattr(vector_db, "_write_manifest", fail_write)

    with pytest.raises(VectorTransactionError):
        vector_db.delete_document(target_id)

    assert raw_vector_snapshot(vector_db) == before_vectors
    assert vector_db.settings.manifest_path.read_bytes() == before_manifest
    assert target.read_text(encoding="utf-8") == "target document"


def test_delete_snapshot_failure_does_not_delete_existing_vectors(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = write_source(manager, "target.txt", "target document")
    unrelated = write_source(manager, "unrelated.txt", "unrelated document")
    assert manager.index_document(target).success
    assert manager.index_document(unrelated).success
    target_id = manager.document_id(target)
    before_vectors = raw_vector_snapshot(manager)
    before_manifest = manager.settings.manifest_path.read_bytes()
    before_sources = {
        path.relative_to(manager.settings.sources_dir): path.read_bytes()
        for path in manager.settings.sources_dir.rglob("*.txt")
    }
    collection = manager._store()._collection
    real_get = collection.get

    def fail_snapshot(*args, **kwargs):
        raise OSError("snapshot failure")

    monkeypatch.setattr(collection, "get", fail_snapshot)

    with pytest.raises(VectorTransactionError):
        manager.delete_document(target_id)

    monkeypatch.setattr(collection, "get", real_get)
    assert raw_vector_snapshot(manager) == before_vectors
    assert manager.settings.manifest_path.read_bytes() == before_manifest
    assert {
        path.relative_to(manager.settings.sources_dir): path.read_bytes()
        for path in manager.settings.sources_dir.rglob("*.txt")
    } == before_sources


def test_delete_cleanup_failure_after_commit_does_not_change_success(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_source(manager, "guide.txt", "committed delete")
    assert manager.index_document(source).success
    document_id = manager.document_id(source)
    real_rmtree = shutil.rmtree

    def cleanup_then_fail(path, *args, **kwargs) -> None:
        real_rmtree(path, *args, **kwargs)
        raise OSError("cleanup failure")

    monkeypatch.setattr(vector_operations.shutil, "rmtree", cleanup_then_fail)

    assert manager.delete_document(document_id) is True
    assert document_id not in manager.manifest().documents
    assert not source.exists()


def test_vector_rollback_failure_still_restores_upload_sources_and_manifest(
    manager: VectorDBManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = manager._store()

    def fail_manifest(_manifest) -> None:
        raise OSError("manifest failure")

    def fail_vector_delete(*, ids) -> None:
        raise OSError("vector rollback failure")

    monkeypatch.setattr(manager, "_write_manifest", fail_manifest)
    monkeypatch.setattr(store, "delete", fail_vector_delete)

    with pytest.raises(VectorTransactionError):
        manager.index_upload_batch(
            (
                UploadedFile(
                    filename="new.txt",
                    content_type="text/plain",
                    content=b"new document",
                ),
            )
        )

    assert not manager.settings.manifest_path.exists()
    assert not list(manager.settings.sources_dir.rglob("*"))
    assert not list(manager.settings.data_dir.glob("rag-upload-*"))


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
