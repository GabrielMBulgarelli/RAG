"""Presentation-neutral workspace orchestration."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from starlette.responses import Response

from modules.application.errors import (
    DocumentNotFoundError,
    IndexStateError,
    InvalidUploadError,
    RuntimeUnavailableError,
    UploadLimitExceededApplicationError,
)
from modules.application.models import (
    CapabilitySnapshot,
    ConversationExportRequest,
    ConversationMessage,
    CorpusSnapshot,
    CorpusState,
    DiagnosticCheck,
    DiagnosticsSnapshot,
    DocumentList,
    DocumentRecord,
    ModelLoadRequest,
    OperationKind,
    QueryDiagnostics,
    QueryRequest,
    QueryResponse,
    RetrievalHit,
    RuntimeSnapshot,
    Source,
    TraceEvent,
    UploadAccepted,
    UploadBatchResult,
    UploadedFile,
)
from modules.application.operation_coordinator import WorkspaceOperationCoordinator
from modules.config import PROJECT_ROOT, Settings, config
from modules.models import IngestionManifest, ManifestDocument, ReconciliationResult
from modules.rag_graph import RAGGraph, make_chat_model
from modules.vector_db import VectorDBManager
from modules.vector_db_operations import (
    UploadLimitExceededError,
    UploadValidationError,
)


@dataclass(frozen=True)
class RuntimeProbeResult:
    reachable: bool
    models: tuple[str, ...]


class _VectorDB(Protocol):
    def setup(self) -> object: ...

    def manifest(self) -> IngestionManifest: ...

    def reconcile_index(self) -> ReconciliationResult: ...

    def chunk_count(self) -> int: ...

    def index_upload_batch(self, files: tuple[UploadedFile, ...]) -> list[ManifestDocument]: ...

    def index_upload_batch_with_manifest(
        self, files: tuple[UploadedFile, ...]
    ) -> tuple[list[ManifestDocument], IngestionManifest]: ...

    def delete_document(self, document_id: str) -> bool: ...

    def delete_document_with_manifest(self, document_id: str) -> IngestionManifest | None: ...


class _Graph(Protocol):
    def process_query(
        self,
        query: str,
        session_id: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]: ...

    def clear(self, session_id: str) -> None: ...


VectorDBFactory = Callable[[], _VectorDB]
GraphFactory = Callable[[_VectorDB, str], _Graph]
RuntimeProbe = Callable[[], RuntimeProbeResult]
BENCHMARK_PREPARATION_COMMAND = "uv run python scripts/prepare_multihop_eval.py --index"


@dataclass(frozen=True)
class BenchmarkPreparationStatus:
    checks: tuple[DiagnosticCheck, ...]

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(check.state == "ready" for check in self.checks)


DatasetReadyProbe = Callable[[], BenchmarkPreparationStatus]
CompletedBenchmarkProbe = Callable[[], Awaitable[bool]]


def _normalize_model_name(name: str) -> str:
    normalized = name.strip()
    return normalized if ":" in normalized else f"{normalized}:latest"


def _default_runtime_probe(settings: Settings) -> RuntimeProbeResult:
    try:
        with urllib.request.urlopen(f"{settings.ollama_base_url}/api/tags", timeout=2) as response:
            payload = json.load(response)
        models = payload.get("models", [])
        return RuntimeProbeResult(
            reachable=True,
            models=tuple(str(item.get("name", "")) for item in models if isinstance(item, dict)),
        )
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TypeError):
        return RuntimeProbeResult(reachable=False, models=())


def _default_graph_factory(vector_db: _VectorDB, chat_model: str) -> _Graph:
    return cast(
        _Graph,
        RAGGraph(
            cast(Any, vector_db),
            llm=make_chat_model(chat_model),
        ),
    )


class WorkspaceService:
    """Concrete async boundary for the synchronous local RAG workspace."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        coordinator: WorkspaceOperationCoordinator | None = None,
        vector_db_factory: VectorDBFactory | None = None,
        graph_factory: GraphFactory | None = None,
        runtime_probe: RuntimeProbe | None = None,
        dataset_ready_probe: DatasetReadyProbe | None = None,
        completed_benchmark_probe: CompletedBenchmarkProbe | None = None,
    ) -> None:
        self.settings = settings or config
        self.coordinator = coordinator or WorkspaceOperationCoordinator()
        self._vector_db_factory = vector_db_factory or (
            lambda: cast(_VectorDB, VectorDBManager(settings=self.settings))
        )
        self._graph_factory = graph_factory or _default_graph_factory
        self._runtime_probe = runtime_probe or (lambda: _default_runtime_probe(self.settings))
        self._dataset_ready_probe = dataset_ready_probe or self._default_dataset_ready_probe
        self._completed_benchmark_probe = completed_benchmark_probe or self._no_completed_benchmark
        self._vector_db: _VectorDB | None = None
        self._graph: _Graph | None = None
        self._active_chat_model: str | None = None
        self._conversations: dict[UUID, list[ConversationMessage]] = {}
        self._latest_public_query: dict[UUID, dict[str, Any]] = {}

    @property
    def active_chat_model(self) -> str | None:
        return self._active_chat_model

    async def start(self) -> None:
        for directory in (
            self.settings.sources_dir,
            self.settings.data_dir,
            self.settings.trace_dir,
            self.settings.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    async def close(self) -> None:
        self._graph = None
        self._vector_db = None
        self._active_chat_model = None
        self._conversations.clear()
        self._latest_public_query.clear()

    def _manager(self) -> _VectorDB:
        if self._vector_db is None:
            self._vector_db = self._vector_db_factory()
        return self._vector_db

    @staticmethod
    def _corpus(manifest: IngestionManifest, *, status: CorpusState = "ready") -> CorpusSnapshot:
        documents = tuple(manifest.documents.values())
        corpus_status: CorpusState = "empty" if not documents and status == "ready" else status
        return CorpusSnapshot(
            document_count=len(documents),
            page_count=sum(document.page_count for document in documents),
            chunk_count=sum(document.chunk_count for document in documents),
            status=corpus_status,
        )

    @staticmethod
    def _document(record: ManifestDocument) -> DocumentRecord:
        fallback = datetime.fromtimestamp(0, timezone.utc)
        return DocumentRecord(
            id=record.document_id,
            filename=record.filename,
            state="indexed",
            size_bytes=record.size_bytes,
            page_count=record.page_count,
            chunk_count=record.chunk_count,
            indexed_at=record.indexed_at,
            updated_at=record.updated_at or record.indexed_at or fallback,
        )

    def _document_list(self, manifest: IngestionManifest) -> DocumentList:
        documents = sorted(
            (self._document(record) for record in manifest.documents.values()),
            key=lambda record: (record.filename.casefold(), record.id),
        )
        return DocumentList(
            documents=documents,
            corpus=self._corpus(manifest),
            active_operation=self.coordinator.snapshot(),
        )

    async def _probe(self) -> RuntimeProbeResult:
        try:
            return await asyncio.to_thread(self._runtime_probe)
        except Exception:
            return RuntimeProbeResult(reachable=False, models=())

    async def _manifest(self) -> IngestionManifest:
        try:
            return await asyncio.to_thread(self._read_manifest_file)
        except Exception as exc:
            raise IndexStateError(reason="manifest_invalid") from exc

    def _read_manifest_file(self) -> IngestionManifest:
        path = self.settings.manifest_path
        if not path.exists():
            return IngestionManifest()
        return IngestionManifest.model_validate_json(path.read_text(encoding="utf-8"))

    async def get_runtime(self) -> RuntimeSnapshot:
        probe, manifest, benchmark_preparation = await asyncio.gather(
            self._probe(), self._manifest(), self._benchmark_preparation()
        )
        return self._runtime_snapshot(
            probe=probe,
            manifest=manifest,
            benchmark_preparation=benchmark_preparation,
        )

    def _runtime_snapshot(
        self,
        *,
        probe: RuntimeProbeResult,
        manifest: IngestionManifest,
        benchmark_preparation: BenchmarkPreparationStatus,
    ) -> RuntimeSnapshot:
        installed = {_normalize_model_name(model) for model in probe.models}
        required = {
            _normalize_model_name(self.settings.embedding_model),
            _normalize_model_name(self.settings.llm_model),
        }
        ready = probe.reachable and required.issubset(installed)
        active = self.coordinator.snapshot()
        loaded = self._graph is not None
        return RuntimeSnapshot(
            state="ready" if loaded else ("not_loaded" if ready else "blocked"),
            configured_chat_model=self.settings.llm_model,
            active_chat_model=self._active_chat_model,
            embedding_model=self.settings.embedding_model,
            available_chat_models=self._available_chat_models(probe.models),
            detail=self._runtime_detail(loaded=loaded, ready=ready),
            capabilities=self._runtime_capabilities(
                probe=probe,
                benchmark_preparation=benchmark_preparation,
                installed=installed,
                ready=ready,
            ),
            active_operation=active,
            corpus=self._corpus(manifest),
        )

    def _available_chat_models(self, models: tuple[str, ...]) -> list[str]:
        embedding_model = _normalize_model_name(self.settings.embedding_model)
        return sorted(
            {
                _normalize_model_name(model)
                for model in models
                if model.strip() and _normalize_model_name(model) != embedding_model
            }
        )

    @staticmethod
    def _runtime_detail(*, loaded: bool, ready: bool) -> str:
        if loaded:
            return "Models are loaded and ready."
        if ready:
            return "Required local models are available."
        return "Start Ollama and install the configured local models."

    def _runtime_capabilities(
        self,
        *,
        probe: RuntimeProbeResult,
        benchmark_preparation: BenchmarkPreparationStatus,
        installed: set[str],
        ready: bool,
    ) -> CapabilitySnapshot:
        loaded = self._graph is not None
        idle = self.coordinator.snapshot() is None
        return CapabilitySnapshot(
            can_query=loaded and idle,
            can_load_models=ready and not loaded and idle,
            can_upload=probe.reachable
            and _normalize_model_name(self.settings.embedding_model) in installed
            and idle,
            can_run_benchmark=benchmark_preparation.ready and loaded and idle,
        )

    def _load_model_blocking(self, chat_model: str) -> tuple[_VectorDB, _Graph]:
        vector_db = self._vector_db_factory()
        vector_db.setup()
        return vector_db, self._graph_factory(vector_db, chat_model)

    async def load_model(self, request: ModelLoadRequest) -> RuntimeSnapshot:
        with self.coordinator.acquire(OperationKind.LOAD_MODEL):
            probe = await self._probe()
            installed = {_normalize_model_name(model) for model in probe.models}
            required = {
                _normalize_model_name(request.chat_model),
                _normalize_model_name(self.settings.embedding_model),
            }
            if not probe.reachable or not required.issubset(installed):
                raise RuntimeUnavailableError(operation="load_model")
            try:
                vector_db, graph = await asyncio.to_thread(
                    self._load_model_blocking, request.chat_model
                )
            except Exception as exc:
                raise RuntimeUnavailableError(operation="load_model") from exc
            self._vector_db = vector_db
            self._graph = graph
            self._active_chat_model = request.chat_model
        return await self.get_runtime()

    @staticmethod
    def _benchmark_manifest(manifest_path: Path) -> IngestionManifest | None:
        try:
            if manifest_path.is_file():
                return IngestionManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
        except (OSError, ValueError):
            return None
        return None

    @staticmethod
    def _benchmark_chunk_count(
        *, multihop_dir: Path, runtime_dir: Path, manifest_path: Path
    ) -> int:
        chroma_dir = runtime_dir / "chroma"
        if not (chroma_dir / "chroma.sqlite3").is_file():
            return 0
        benchmark_settings = config.model_copy(
            update={
                "sources_dir": multihop_dir / "corpus",
                "data_dir": runtime_dir,
                "chroma_dir": chroma_dir,
                "manifest_path": manifest_path,
            }
        )
        try:
            return VectorDBManager(settings=benchmark_settings).chunk_count()
        except Exception:
            return 0

    @staticmethod
    def _benchmark_preparation_status(
        *, files_ready: bool, manifest: IngestionManifest | None, chunk_count: int
    ) -> BenchmarkPreparationStatus:
        manifest_ready = manifest is not None and bool(manifest.documents)
        command_detail = f"Run {BENCHMARK_PREPARATION_COMMAND}."
        return BenchmarkPreparationStatus(
            checks=(
                DiagnosticCheck(
                    area="evaluation",
                    name="Benchmark files",
                    state="ready" if files_ready else "blocked",
                    detail=(
                        "Benchmark cases and source mapping are available."
                        if files_ready
                        else f"Benchmark cases or source mapping are missing. {command_detail}"
                    ),
                ),
                DiagnosticCheck(
                    area="evaluation",
                    name="Benchmark manifest",
                    state="ready" if manifest_ready else "blocked",
                    detail=(
                        f"Benchmark manifest contains {len(manifest.documents)} documents."
                        if manifest_ready and manifest is not None
                        else f"Benchmark manifest is missing or empty. {command_detail}"
                    ),
                ),
                DiagnosticCheck(
                    area="evaluation",
                    name="Benchmark index",
                    state="ready" if chunk_count > 0 else "blocked",
                    detail=(
                        f"Benchmark index contains {chunk_count} chunks."
                        if chunk_count > 0
                        else f"Benchmark index is missing or empty. {command_detail}"
                    ),
                ),
            )
        )

    @classmethod
    def _default_dataset_ready_probe(cls) -> BenchmarkPreparationStatus:
        multihop_dir = PROJECT_ROOT / "evals" / "multihop"
        runtime_dir = PROJECT_ROOT / "evals" / "runtime"
        required_files = (multihop_dir / "cases.jsonl", multihop_dir / "source_map.json")
        manifest_path = runtime_dir / "manifest.json"
        return cls._benchmark_preparation_status(
            files_ready=all(path.is_file() for path in required_files),
            manifest=cls._benchmark_manifest(manifest_path),
            chunk_count=cls._benchmark_chunk_count(
                multihop_dir=multihop_dir,
                runtime_dir=runtime_dir,
                manifest_path=manifest_path,
            ),
        )

    async def _benchmark_preparation(self) -> BenchmarkPreparationStatus:
        try:
            return await asyncio.to_thread(self._dataset_ready_probe)
        except Exception:
            return BenchmarkPreparationStatus(
                checks=(
                    DiagnosticCheck(
                        area="evaluation",
                        name="Benchmark preparation",
                        state="blocked",
                        detail=(
                            "Benchmark preparation could not be verified. "
                            f"Run {BENCHMARK_PREPARATION_COMMAND}."
                        ),
                    ),
                )
            )

    @staticmethod
    async def _no_completed_benchmark() -> bool:
        return False

    async def get_diagnostics(self) -> DiagnosticsSnapshot:
        probe = await self._probe()
        installed = {_normalize_model_name(model) for model in probe.models}
        runtime_checks = [
            DiagnosticCheck(
                area="runtime",
                name="Ollama connectivity",
                state="ready" if probe.reachable else "blocked",
                detail="Reachable." if probe.reachable else "Start the configured Ollama service.",
            ),
            DiagnosticCheck(
                area="runtime",
                name="Chat model",
                state=(
                    "ready"
                    if _normalize_model_name(self.settings.llm_model) in installed
                    else "blocked"
                ),
                detail="Configured chat model is available."
                if _normalize_model_name(self.settings.llm_model) in installed
                else "Install the configured chat model.",
            ),
            DiagnosticCheck(
                area="runtime",
                name="Embedding model",
                state=(
                    "ready"
                    if _normalize_model_name(self.settings.embedding_model) in installed
                    else "blocked"
                ),
                detail="Configured embedding model is available."
                if _normalize_model_name(self.settings.embedding_model) in installed
                else "Install the configured embedding model.",
            ),
            DiagnosticCheck(
                area="runtime",
                name="Model initialization",
                state="ready" if self._graph is not None else "not_loaded",
                detail="Loaded for this session."
                if self._graph is not None
                else "Load a chat model when needed.",
            ),
        ]
        try:
            manager = self._manager()
            manifest, reconciliation, chunks = await asyncio.to_thread(
                self._index_diagnostics, manager
            )
            index_checks = self._index_checks(manifest, reconciliation, chunks)
        except IndexStateError as exc:
            index_checks = [self._index_error_check(exc)]
        except Exception:
            index_checks = [
                DiagnosticCheck(
                    area="index",
                    name="Index diagnostics",
                    state="error",
                    detail="The local index could not be inspected.",
                )
            ]
        benchmark_preparation = await self._benchmark_preparation()
        try:
            completed_benchmark_available = await self._completed_benchmark_probe()
        except Exception:
            completed_benchmark_available = False
        evaluation_checks = [
            *benchmark_preparation.checks,
            DiagnosticCheck(
                area="evaluation",
                name="Latest completed Full RAG Benchmark artifact",
                state=("ready" if completed_benchmark_available else "not_loaded"),
                detail="A complete Full RAG Benchmark artifact is available."
                if completed_benchmark_available
                else "No complete Full RAG Benchmark artifact is stored.",
            ),
        ]
        all_checks = [*runtime_checks, *index_checks, *evaluation_checks]
        states = {check.state for check in all_checks}
        state = (
            "error"
            if "error" in states
            else "blocked"
            if "blocked" in states
            else "review"
            if "review" in states
            else "ready"
        )
        return DiagnosticsSnapshot(
            state=state,
            title="Workspace diagnostics",
            detail="Review unavailable or inconsistent local dependencies."
            if state != "ready"
            else "All workspace checks are ready.",
            runtime_checks=runtime_checks,
            index_checks=index_checks,
            evaluation_checks=evaluation_checks,
            active_operation=self.coordinator.snapshot(),
            stale=self.coordinator.snapshot() is not None,
        )

    @classmethod
    def _index_diagnostics(
        cls,
        manager: _VectorDB,
    ) -> tuple[IngestionManifest, ReconciliationResult, int]:
        manifest = cls._manager_manifest(manager)
        try:
            return manifest, manager.reconcile_index(), manager.chunk_count()
        except Exception as exc:
            raise IndexStateError(reason="collection_unavailable") from exc

    @staticmethod
    def _manager_manifest(manager: _VectorDB) -> IngestionManifest:
        try:
            return manager.manifest()
        except Exception as exc:
            raise IndexStateError(reason="manifest_invalid") from exc

    @staticmethod
    def _index_error_check(error: IndexStateError) -> DiagnosticCheck:
        if error.details["reason"] == "manifest_invalid":
            return DiagnosticCheck(
                area="index",
                name="Manifest",
                state="error",
                detail=(
                    "The manifest is malformed or unreadable. Restore or remove it, "
                    "then rebuild the local index."
                ),
            )
        return DiagnosticCheck(
            area="index",
            name="Chroma collection",
            state="error",
            detail="Chroma data could not be read. Rebuild the local index.",
        )

    @staticmethod
    def _index_checks(
        manifest: IngestionManifest,
        reconciliation: ReconciliationResult,
        chunks: int,
    ) -> list[DiagnosticCheck]:
        checks = WorkspaceService._core_index_checks(
            manifest=manifest,
            chunks=chunks,
            reconciliation=reconciliation,
        )
        checks.extend(WorkspaceService._reconciliation_checks(reconciliation))
        return checks

    @staticmethod
    def _core_index_checks(
        *,
        manifest: IngestionManifest,
        chunks: int,
        reconciliation: ReconciliationResult,
    ) -> list[DiagnosticCheck]:
        expected_chunks = sum(document.chunk_count for document in manifest.documents.values())
        missing_collection = expected_chunks > 0 and chunks == 0
        missing_chunks = len(reconciliation.missing_chunk_ids)
        incompatible_documents = len(reconciliation.incompatible_document_ids)
        return [
            WorkspaceService._collection_diagnostic(
                chunks=chunks, missing_collection=missing_collection
            ),
            DiagnosticCheck(
                area="index",
                name="Manifest",
                state="ready",
                detail=f"Valid; {len(manifest.documents)} documents.",
            ),
            WorkspaceService._missing_chunks_diagnostic(missing_chunks),
            WorkspaceService._configuration_diagnostic(incompatible_documents),
        ]

    @staticmethod
    def _collection_diagnostic(*, chunks: int, missing_collection: bool) -> DiagnosticCheck:
        return DiagnosticCheck(
            area="index",
            name="Chroma collection",
            state="blocked" if missing_collection else "ready",
            detail=(
                "No Chroma data was found for the manifest. Rebuild the local index."
                if missing_collection
                else f"{chunks} chunks."
            ),
        )

    @staticmethod
    def _missing_chunks_diagnostic(missing_chunks: int) -> DiagnosticCheck:
        return DiagnosticCheck(
            area="index",
            name="Indexed chunks",
            state="blocked" if missing_chunks else "ready",
            detail=(
                f"{missing_chunks} manifest "
                f"{'chunk is' if missing_chunks == 1 else 'chunks are'} missing from "
                "Chroma. Rebuild the local index."
                if missing_chunks
                else "All manifest chunks are present."
            ),
        )

    @staticmethod
    def _configuration_diagnostic(incompatible_documents: int) -> DiagnosticCheck:
        return DiagnosticCheck(
            area="index",
            name="Index configuration",
            state="blocked" if incompatible_documents else "ready",
            detail=(
                f"{incompatible_documents} "
                f"{'document uses' if incompatible_documents == 1 else 'documents use'} "
                "different index settings. Rebuild the local index."
                if incompatible_documents
                else "Index settings match the current configuration."
            ),
        )

    @staticmethod
    def _reconciliation_checks(
        reconciliation: ReconciliationResult,
    ) -> list[DiagnosticCheck]:
        values = [
            ("Orphan Chroma chunks", reconciliation.orphan_chunk_ids, "review"),
            ("Duplicate IDs", reconciliation.duplicate_chunk_ids, "blocked"),
            ("Missing source files", reconciliation.missing_source_files, "review"),
        ]
        return [
            DiagnosticCheck(
                area="index",
                name=name,
                state=cast(Any, failure_state if problems else "ready"),
                detail=f"{len(problems)} found.",
            )
            for name, problems, failure_state in values
        ]

    async def list_documents(self) -> DocumentList:
        try:
            manifest = await asyncio.to_thread(self._manager_manifest, self._manager())
        except IndexStateError:
            raise
        except Exception as exc:
            raise RuntimeUnavailableError(operation="list_documents") from exc
        return self._document_list(manifest)

    async def upload_documents(self, files: tuple[UploadedFile, ...]) -> UploadBatchResult:
        with self.coordinator.acquire(OperationKind.INDEX_DOCUMENTS):
            manager = self._manager()
            try:
                accepted_records, manifest = await asyncio.to_thread(
                    manager.index_upload_batch_with_manifest,
                    files,
                )
                self._graph = None
                self._active_chat_model = None
            except UploadLimitExceededError as exc:
                raise UploadLimitExceededApplicationError() from exc
            except UploadValidationError as exc:
                raise InvalidUploadError() from exc
            except Exception as exc:
                raise RuntimeUnavailableError(operation="upload_documents") from exc
        listing = self._document_list(manifest)
        return UploadBatchResult(
            accepted=[
                UploadAccepted(filename=record.filename, document_id=record.document_id)
                for record in accepted_records
            ],
            documents=listing.documents,
            corpus=listing.corpus,
        )

    async def delete_document(self, document_id: str) -> DocumentList:
        with self.coordinator.acquire(OperationKind.DELETE_DOCUMENT):
            manager = self._manager()
            try:
                manifest = await asyncio.to_thread(
                    manager.delete_document_with_manifest,
                    document_id,
                )
                if manifest is None:
                    raise DocumentNotFoundError()
                self._graph = None
                self._active_chat_model = None
            except DocumentNotFoundError:
                raise
            except Exception as exc:
                raise RuntimeUnavailableError(operation="delete_document") from exc
        return self._document_list(manifest)

    @staticmethod
    def _answer_state(result: dict[str, Any]) -> str:
        evidence = str(result.get("evidence_status", ""))
        route = str(result.get("route", ""))
        if route == "unavailable":
            return "unavailable"
        if evidence == "sufficient":
            return "supported"
        if evidence in {"partial", "limited"}:
            return "limited"
        if evidence in {"insufficient", "none"}:
            return "abstention"
        return "completed"

    async def query(self, request: QueryRequest) -> QueryResponse:
        with self.coordinator.acquire(OperationKind.QUERY):
            graph = self._graph
            if graph is None:
                raise RuntimeUnavailableError(operation="query")
            try:
                raw = await asyncio.to_thread(
                    graph.process_query, request.question, str(request.session_id)
                )
                response = self._query_response(request, raw)
            except Exception as exc:
                if isinstance(exc, RuntimeUnavailableError):
                    raise
                raise RuntimeUnavailableError(operation="query") from exc
            user_message = ConversationMessage(
                id=uuid4(),
                role="user",
                content=request.question,
                created_at=response.message.created_at,
            )
            self._conversations.setdefault(request.session_id, []).extend(
                [user_message, response.message]
            )
            self._latest_public_query[request.session_id] = {
                "answer_state": response.answer_state,
                "sources": [item.model_dump(mode="json") for item in response.sources],
                "trace": [item.model_dump(mode="json") for item in response.trace],
                "diagnostics": response.diagnostics.model_dump(mode="json"),
            }
            return response

    @staticmethod
    def _query_response(request: QueryRequest, raw: dict[str, Any]) -> QueryResponse:
        sources = [
            Source(
                label=str(item.get("label", "")),
                filename=str(item.get("filename", "")),
                page=item.get("page"),
                excerpt=str(item.get("excerpt", "")),
            )
            for item in raw.get("sources", [])
        ]
        hits = [
            RetrievalHit(
                chunk_id=str(item.get("chunk_id", "")),
                filename=str(item.get("filename", "")),
                page=item.get("page"),
                semantic_score=item.get("semantic_score"),
                sparse_score=item.get("sparse_score"),
                fused_score=item.get("fused_score"),
                selection_score=item.get("selection_score"),
                matched_subqueries=list(item.get("subqueries", [])),
            )
            for item in raw.get("retrieval_hits", [])
        ]
        trace = [
            TraceEvent(
                stage=str(item.get("stage", "")),
                decision=str(item.get("decision") or ""),
                retrieved_count=item.get("retrieved_count", item.get("candidate_count")),
                fused_count=item.get("fused_count"),
                selected_count=item.get("selected_count"),
                retry_count=int(item.get("retry_count", 0)),
                llm_calls=int(item.get("llm_calls", 0)),
                termination=str(item.get("termination") or ""),
                duration_ms=item.get("duration_ms"),
            )
            for item in raw.get("trace", [])
        ]
        evidence = str(raw.get("evidence_status", ""))
        validation_value = raw.get("validation")
        validation = validation_value if isinstance(validation_value, dict) else {}
        citation_validation = (
            "valid"
            if validation.get("is_valid") is True
            else ("invalid" if validation else "not_reported")
        )
        diagnostics = QueryDiagnostics(
            route=str(raw.get("route", "")),
            retrieval_strategy=str(raw.get("strategy", "")),
            subqueries=list(raw.get("subqueries", [])),
            retry_count=int(raw.get("retry_count", 0)),
            evidence_state=evidence,
            conflict_state="conflict" if raw.get("conflict") else "none",
            citation_validation=citation_validation,
        )
        return QueryResponse(
            session_id=request.session_id,
            message=ConversationMessage(
                id=uuid4(),
                role="assistant",
                content=str(raw.get("answer", "")),
                created_at=datetime.now(timezone.utc),
            ),
            answer_state=cast(Any, WorkspaceService._answer_state(raw)),
            sources=sources,
            retrieval_hits=hits,
            trace=trace,
            diagnostics=diagnostics,
        )

    async def clear_conversation(self, session_id: UUID) -> None:
        with self.coordinator.acquire(OperationKind.QUERY):
            graph = self._graph
            if graph is not None:
                try:
                    await asyncio.to_thread(graph.clear, str(session_id))
                except Exception as exc:
                    raise RuntimeUnavailableError(operation="clear_conversation") from exc
            self._conversations.pop(session_id, None)
            self._latest_public_query.pop(session_id, None)

    async def export_conversation(self, request: ConversationExportRequest) -> Response:
        payload = {
            "session_id": str(request.session_id),
            "messages": [
                message.model_dump(mode="json")
                for message in self._conversations.get(request.session_id, [])
            ],
            "latest_query": self._latest_public_query.get(request.session_id),
        }
        return Response(
            json.dumps(payload, ensure_ascii=False),
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="conversation-{request.session_id}.json"'
                )
            },
        )
