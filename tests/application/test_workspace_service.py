import asyncio
import json
import threading
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from modules.api.dependencies import UploadedFile as ApiUploadedFile
from modules.application.errors import (
    DocumentNotFoundError,
    InvalidUploadError,
    OperationBusyError,
    RuntimeUnavailableError,
    UploadLimitExceededApplicationError,
)
from modules.application.models import (
    ConversationExportRequest,
    ModelLoadRequest,
    OperationKind,
    QueryRequest,
    UploadedFile,
)
from modules.application.operation_coordinator import WorkspaceOperationCoordinator
from modules.application.workspace_service import RuntimeProbeResult, WorkspaceService
from modules.config import Settings
from modules.models import IngestionManifest, ManifestDocument, ReconciliationResult
from modules.vector_db_operations import (
    UploadLimitExceededError,
    UploadValidationError,
    VectorTransactionError,
)


class FakeVectorDB:
    def __init__(
        self,
        *,
        setup_error: Exception | None = None,
        reconciliation: ReconciliationResult | None = None,
    ) -> None:
        self.setup_error = setup_error
        self.setup_calls = 0
        self._manifest = IngestionManifest()
        self.reconciliation = reconciliation or ReconciliationResult()
        self.worker_threads: list[int] = []

    def setup(self) -> object:
        self.setup_calls += 1
        if self.setup_error is not None:
            raise self.setup_error
        return object()

    def manifest(self) -> IngestionManifest:
        return self._manifest

    def reconcile_index(self) -> ReconciliationResult:
        return self.reconciliation

    def chunk_count(self) -> int:
        return sum(record.chunk_count for record in self._manifest.documents.values())

    def index_upload_batch(self, files: tuple[UploadedFile, ...]) -> list[ManifestDocument]:
        self.worker_threads.append(threading.get_ident())
        records = []
        now = datetime.now(timezone.utc)
        for index, file in enumerate(files):
            document_id = f"document-{index}"
            record = ManifestDocument(
                document_id=document_id,
                relative_path=f"{document_id}/{file.filename}",
                filename=file.filename,
                content_hash=f"hash-{index}",
                chunk_ids=[f"chunk-{index}"],
                page_count=1,
                chunk_count=1,
                embedding_model="embedding",
                chunk_size=700,
                chunk_overlap=100,
                size_bytes=len(file.content),
                indexed_at=now,
                updated_at=now,
            )
            self._manifest.documents[document_id] = record
            records.append(record)
        return records

    def index_upload_batch_with_manifest(
        self, files: tuple[UploadedFile, ...]
    ) -> tuple[list[ManifestDocument], IngestionManifest]:
        records = self.index_upload_batch(files)
        return records, self._manifest.model_copy(deep=True)

    def delete_document(self, document_id: str) -> bool:
        self.worker_threads.append(threading.get_ident())
        return self._manifest.documents.pop(document_id, None) is not None

    def delete_document_with_manifest(self, document_id: str) -> IngestionManifest | None:
        if not self.delete_document(document_id):
            return None
        return self._manifest.model_copy(deep=True)


class FakeGraph:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {}
        self.cleared: list[str] = []
        self.worker_threads: list[int] = []

    def process_query(
        self,
        query: str,
        session_id: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.worker_threads.append(threading.get_ident())
        return self.result

    def clear(self, session_id: str) -> None:
        self.worker_threads.append(threading.get_ident())
        self.cleared.append(session_id)


class UploadFailureVectorDB(FakeVectorDB):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def index_upload_batch(self, files: tuple[UploadedFile, ...]) -> list[ManifestDocument]:
        raise self.error

    def index_upload_batch_with_manifest(
        self, files: tuple[UploadedFile, ...]
    ) -> tuple[list[ManifestDocument], IngestionManifest]:
        raise self.error


class CommittedSnapshotVectorDB(FakeVectorDB):
    def manifest(self) -> IngestionManifest:
        raise AssertionError("post-commit manifest reads are forbidden")

    def index_upload_batch(self, files: tuple[UploadedFile, ...]) -> list[ManifestDocument]:
        raise AssertionError("service must request the committed manifest")

    def index_upload_batch_with_manifest(
        self, files: tuple[UploadedFile, ...]
    ) -> tuple[list[ManifestDocument], IngestionManifest]:
        records = super().index_upload_batch(files)
        return records, self._manifest.model_copy(deep=True)

    def delete_document(self, document_id: str) -> bool:
        raise AssertionError("service must request the committed manifest")

    def delete_document_with_manifest(self, document_id: str) -> IngestionManifest | None:
        FakeVectorDB.delete_document(self, document_id)
        return self._manifest.model_copy(deep=True)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        sources_dir=tmp_path / "sources",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        manifest_path=tmp_path / "data" / "manifest.json",
        trace_dir=tmp_path / "data" / "traces",
        logs_dir=tmp_path / "logs",
    )


def test_uploaded_file_is_an_immutable_application_value_object() -> None:
    uploaded = UploadedFile(filename="guide.txt", content_type="text/plain", content=b"guide")

    assert ApiUploadedFile is UploadedFile
    with pytest.raises(FrozenInstanceError):
        setattr(uploaded, "filename", "changed.txt")


def test_start_only_creates_local_directories_and_close_is_idempotent(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    factory_calls = 0

    def vector_db_factory() -> FakeVectorDB:
        nonlocal factory_calls
        factory_calls += 1
        return FakeVectorDB()

    service = WorkspaceService(
        settings=settings,
        vector_db_factory=vector_db_factory,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(reachable=True, models=()),
    )

    asyncio.run(service.start())
    asyncio.run(service.close())
    asyncio.run(service.close())

    assert factory_calls == 0
    assert settings.sources_dir.is_dir()
    assert settings.data_dir.is_dir()
    assert settings.trace_dir.is_dir()
    assert settings.logs_dir.is_dir()


def test_runtime_capabilities_follow_readiness_and_coordinator(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    coordinator = WorkspaceOperationCoordinator()
    probe = RuntimeProbeResult(
        reachable=True,
        models=(settings.llm_model, settings.embedding_model, "other:latest"),
    )
    service = WorkspaceService(
        settings=settings,
        coordinator=coordinator,
        vector_db_factory=FakeVectorDB,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: probe,
    )

    runtime = asyncio.run(service.get_runtime())

    assert runtime.state == "not_loaded"
    assert runtime.available_chat_models == [
        "other:latest",
        settings.llm_model,
    ]
    assert runtime.capabilities.can_load_models is True
    assert runtime.capabilities.can_query is False

    with coordinator.acquire(OperationKind.BENCHMARK):
        busy = asyncio.run(service.get_runtime())

    assert busy.active_operation is not None
    assert busy.active_operation.kind is OperationKind.BENCHMARK
    assert busy.capabilities.model_dump() == {
        "can_query": False,
        "can_load_models": False,
        "can_upload": False,
        "can_run_benchmark": False,
    }


def test_runtime_keeps_benchmark_disabled_when_no_executor_is_configured(
    tmp_path: Path,
) -> None:
    # Given a ready model and indexed corpus without a benchmark executor
    settings = make_settings(tmp_path)
    manager = FakeVectorDB()
    manager.index_upload_batch(
        (UploadedFile(filename="guide.txt", content_type="text/plain", content=b"guide"),)
    )
    service = WorkspaceService(
        settings=settings,
        benchmark_available=False,
        vector_db_factory=lambda: manager,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(
            reachable=True,
            models=(settings.llm_model, settings.embedding_model),
        ),
    )

    # When runtime capability is evaluated
    asyncio.run(service.load_model(ModelLoadRequest(chat_model=settings.llm_model)))
    runtime = asyncio.run(service.get_runtime())

    # Then querying remains available while benchmarking does not
    assert runtime.state == "ready"
    assert runtime.capabilities.can_query is True
    assert runtime.capabilities.can_run_benchmark is False


def test_loaded_runtime_can_benchmark_embedded_corpus_without_workspace_documents(
    tmp_path: Path,
) -> None:
    # Arrange
    settings = make_settings(tmp_path)
    service = WorkspaceService(
        settings=settings,
        vector_db_factory=FakeVectorDB,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(
            reachable=True,
            models=(settings.llm_model, settings.embedding_model),
        ),
    )

    # Act
    asyncio.run(service.load_model(ModelLoadRequest(chat_model=settings.llm_model)))
    runtime = asyncio.run(service.get_runtime())

    # Then the embedded benchmark is independent from workspace documents
    assert runtime.corpus.document_count == 0
    assert runtime.capabilities.can_run_benchmark is True
    assert service.active_chat_model == settings.llm_model


def test_model_load_publishes_only_after_full_success(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    managers = [
        FakeVectorDB(setup_error=RuntimeError("private runtime failure")),
        FakeVectorDB(),
    ]
    graphs: list[FakeGraph] = []

    def vector_db_factory() -> FakeVectorDB:
        return managers.pop(0)

    def graph_factory(_vector_db: object, _model: str) -> FakeGraph:
        graph = FakeGraph()
        graphs.append(graph)
        return graph

    service = WorkspaceService(
        settings=settings,
        vector_db_factory=vector_db_factory,
        graph_factory=graph_factory,
        runtime_probe=lambda: RuntimeProbeResult(
            reachable=True,
            models=(settings.llm_model, settings.embedding_model),
        ),
    )

    with pytest.raises(RuntimeUnavailableError) as error:
        asyncio.run(service.load_model(ModelLoadRequest(chat_model=settings.llm_model)))

    assert error.value.code == "runtime_unavailable"
    assert "private runtime failure" not in error.value.message
    assert asyncio.run(service.get_runtime()).active_chat_model is None
    assert service.coordinator.snapshot() is None

    loaded = asyncio.run(service.load_model(ModelLoadRequest(chat_model=settings.llm_model)))

    assert loaded.state == "ready"
    assert loaded.active_chat_model == settings.llm_model
    assert loaded.capabilities.can_query is True
    assert len(graphs) == 1


def test_diagnostics_cover_runtime_index_and_evaluation_states(tmp_path: Path) -> None:
    async def no_completed_benchmark() -> bool:
        return False

    async def completed_benchmark() -> bool:
        return True

    settings = make_settings(tmp_path)
    review_manager = FakeVectorDB(reconciliation=ReconciliationResult(orphan_chunk_ids=["orphan"]))
    blocked = WorkspaceService(
        settings=settings,
        vector_db_factory=lambda: review_manager,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(reachable=False, models=()),
        dataset_ready_probe=lambda: False,
        completed_benchmark_probe=no_completed_benchmark,
    )

    blocked_snapshot = asyncio.run(blocked.get_diagnostics())

    assert blocked_snapshot.state == "blocked"
    assert {check.area for check in blocked_snapshot.runtime_checks} == {"runtime"}
    assert any(check.state == "review" for check in blocked_snapshot.index_checks)
    assert blocked_snapshot.evaluation_checks[0].state == "blocked"

    ready = WorkspaceService(
        settings=settings,
        vector_db_factory=FakeVectorDB,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(
            reachable=True,
            models=(settings.llm_model, settings.embedding_model),
        ),
        dataset_ready_probe=lambda: True,
        completed_benchmark_probe=completed_benchmark,
    )

    ready_snapshot = asyncio.run(ready.get_diagnostics())

    assert ready_snapshot.state == "ready"
    assert all(check.state in {"ready", "not_loaded"} for check in ready_snapshot.runtime_checks)
    assert all(check.state == "ready" for check in ready_snapshot.index_checks)
    assert all(check.state == "ready" for check in ready_snapshot.evaluation_checks)
    assert ready_snapshot.evaluation_checks[1].name == (
        "Latest completed Full RAG Benchmark artifact"
    )
    assert ready_snapshot.evaluation_checks[1].detail == (
        "A complete Full RAG Benchmark artifact is available."
    )


def test_document_upload_delete_mapping_worker_thread_and_busy_guard(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    manager = FakeVectorDB()
    coordinator = WorkspaceOperationCoordinator()
    service = WorkspaceService(
        settings=settings,
        coordinator=coordinator,
        vector_db_factory=lambda: manager,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(reachable=True, models=()),
    )
    main_thread = threading.get_ident()
    upload = asyncio.run(
        service.upload_documents((UploadedFile("guide.txt", "text/plain", b"workspace guide"),))
    )

    assert upload.accepted[0].filename == "guide.txt"
    assert upload.accepted[0].document_id == "document-0"
    assert upload.documents[0].size_bytes == len(b"workspace guide")
    assert upload.corpus.document_count == 1
    assert manager.worker_threads[-1] != main_thread

    with coordinator.acquire(OperationKind.BENCHMARK):
        with pytest.raises(OperationBusyError):
            asyncio.run(
                service.upload_documents((UploadedFile("blocked.txt", "text/plain", b"blocked"),))
            )

    deleted = asyncio.run(service.delete_document("document-0"))

    assert deleted.documents == []
    assert deleted.corpus.status == "empty"
    assert manager.worker_threads[-1] != main_thread

    with pytest.raises(DocumentNotFoundError) as error:
        asyncio.run(service.delete_document("missing"))
    assert error.value.code == "document_not_found"


def test_document_mutations_use_committed_snapshot_and_invalidate_graph_immediately(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    manager = CommittedSnapshotVectorDB()
    service = WorkspaceService(
        settings=settings,
        vector_db_factory=lambda: manager,
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(
            reachable=True,
            models=(settings.llm_model, settings.embedding_model),
        ),
    )
    asyncio.run(service.load_model(ModelLoadRequest(chat_model=settings.llm_model)))

    uploaded = asyncio.run(
        service.upload_documents((UploadedFile("guide.txt", "text/plain", b"committed guide"),))
    )

    assert uploaded.corpus.document_count == 1
    assert asyncio.run(service.get_runtime()).active_chat_model is None

    service._graph = FakeGraph()
    service._active_chat_model = settings.llm_model
    deleted = asyncio.run(service.delete_document(uploaded.accepted[0].document_id))

    assert deleted.corpus.status == "empty"
    assert asyncio.run(service.get_runtime()).active_chat_model is None


@pytest.mark.parametrize(
    ("core_error", "application_error", "code"),
    [
        (UploadValidationError("private filename"), InvalidUploadError, "invalid_upload"),
        (
            UploadLimitExceededError("private byte count"),
            UploadLimitExceededApplicationError,
            "upload_limit_exceeded",
        ),
        (
            VectorTransactionError("private database failure"),
            RuntimeUnavailableError,
            "runtime_unavailable",
        ),
    ],
)
def test_upload_failures_are_normalized_without_private_details(
    tmp_path: Path,
    core_error: Exception,
    application_error: type[Exception],
    code: str,
) -> None:
    service = WorkspaceService(
        settings=make_settings(tmp_path),
        vector_db_factory=lambda: UploadFailureVectorDB(core_error),
        graph_factory=lambda _vector_db, _model: FakeGraph(),
        runtime_probe=lambda: RuntimeProbeResult(reachable=True, models=()),
    )

    with pytest.raises(application_error) as error:
        asyncio.run(service.upload_documents((UploadedFile("guide.txt", "text/plain", b"valid"),)))

    assert getattr(error.value, "code") == code
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    ("evidence_status", "answer_state"),
    [
        ("sufficient", "supported"),
        ("limited", "limited"),
        ("insufficient", "abstention"),
    ],
)
def test_query_maps_public_observability_without_inventing_scores(
    tmp_path: Path,
    evidence_status: str,
    answer_state: str,
) -> None:
    settings = make_settings(tmp_path)
    session_id = uuid4()
    result = {
        "answer": "The indexed guide answers this [C1].",
        "route": "simple_search",
        "strategy": "hybrid",
        "retry_count": 1,
        "evidence_status": evidence_status,
        "sources": [
            {
                "label": "C1",
                "chunk_id": "chunk-1",
                "filename": "guide.txt",
                "page": 1,
                "excerpt": "indexed guide",
            }
        ],
        "subqueries": ["indexed guide"],
        "retrieval_hits": [
            {
                "chunk_id": "chunk-1",
                "filename": "guide.txt",
                "page": 1,
                "semantic_score": None,
                "sparse_score": 0.5,
                "fused_score": None,
                "selection_score": 0.8,
                "subqueries": ["indexed guide"],
            }
        ],
        "trace": [
            {
                "stage": "retrieve",
                "decision": "hybrid",
                "candidate_count": 4,
                "retry_count": 1,
                "llm_calls": 0,
                "duration_ms": 1.5,
            }
        ],
        "conflict": True,
    }
    graph = FakeGraph(result)
    service = WorkspaceService(
        settings=settings,
        vector_db_factory=FakeVectorDB,
        graph_factory=lambda _vector_db, _model: graph,
        runtime_probe=lambda: RuntimeProbeResult(
            reachable=True,
            models=(settings.llm_model, settings.embedding_model),
        ),
    )
    asyncio.run(service.load_model(ModelLoadRequest(chat_model=settings.llm_model)))

    response = asyncio.run(
        service.query(QueryRequest(session_id=session_id, question="What does it say?"))
    )

    assert response.answer_state == answer_state
    assert response.sources[0].label == "C1"
    assert response.retrieval_hits[0].semantic_score is None
    assert response.retrieval_hits[0].fused_score is None
    assert response.trace[0].retrieved_count == 4
    assert response.diagnostics.citation_validation == "not_reported"
    assert response.diagnostics.conflict_state == "conflict"
    assert len(graph.worker_threads) == 1
    assert graph.worker_threads[0] != threading.get_ident()


def test_unloaded_query_clear_busy_and_sanitized_export(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    coordinator = WorkspaceOperationCoordinator()
    graph = FakeGraph(
        {
            "answer": "Public answer.",
            "route": "catalog",
            "strategy": "none",
            "retry_count": 0,
            "evidence_status": "sufficient",
            "validation": {"is_valid": True, "private_prompt": "must not export"},
        }
    )
    service = WorkspaceService(
        settings=settings,
        coordinator=coordinator,
        vector_db_factory=FakeVectorDB,
        graph_factory=lambda _vector_db, _model: graph,
        runtime_probe=lambda: RuntimeProbeResult(
            reachable=True,
            models=(settings.llm_model, settings.embedding_model),
        ),
    )
    session_id = uuid4()

    with pytest.raises(RuntimeUnavailableError):
        asyncio.run(service.query(QueryRequest(session_id=session_id, question="Before loading?")))

    asyncio.run(service.load_model(ModelLoadRequest(chat_model=settings.llm_model)))
    asyncio.run(service.query(QueryRequest(session_id=session_id, question="Loaded question")))
    exported = asyncio.run(
        service.export_conversation(ConversationExportRequest(session_id=session_id))
    )
    body = bytes(exported.body).decode("utf-8")

    assert "Public answer." in body
    assert "Loaded question" in body
    assert "private_prompt" not in body
    assert "attachment;" in exported.headers["content-disposition"]

    with coordinator.acquire(OperationKind.BENCHMARK):
        with pytest.raises(OperationBusyError):
            asyncio.run(service.clear_conversation(session_id))

    asyncio.run(service.clear_conversation(session_id))
    cleared = asyncio.run(
        service.export_conversation(ConversationExportRequest(session_id=session_id))
    )

    assert json.loads(bytes(cleared.body))["messages"] == []
    assert graph.cleared == [str(session_id)]
