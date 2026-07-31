"""Presentation-neutral workspace orchestration."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from starlette.responses import Response

from modules.application.errors import (
    DocumentNotFoundError,
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
from modules.evaluation_models import is_standard_benchmark_summary
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


@dataclass(frozen=True)
class EvaluationProbeResult:
    dataset_ready: bool
    latest_compatible: bool


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
EvaluationProbe = Callable[[], EvaluationProbeResult]


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
        evaluation_probe: EvaluationProbe | None = None,
        benchmark_available: bool = True,
    ) -> None:
        self.settings = settings or config
        self.coordinator = coordinator or WorkspaceOperationCoordinator()
        self._vector_db_factory = vector_db_factory or (
            lambda: cast(_VectorDB, VectorDBManager(settings=self.settings))
        )
        self._graph_factory = graph_factory or _default_graph_factory
        self._runtime_probe = runtime_probe or (lambda: _default_runtime_probe(self.settings))
        self._evaluation_probe = evaluation_probe or self._default_evaluation_probe
        self._benchmark_available = benchmark_available
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
        except Exception:
            return IngestionManifest()

    def _read_manifest_file(self) -> IngestionManifest:
        path = self.settings.manifest_path
        if not path.is_file():
            return IngestionManifest()
        return IngestionManifest.model_validate_json(path.read_text(encoding="utf-8"))

    async def get_runtime(self) -> RuntimeSnapshot:
        probe, manifest = await asyncio.gather(self._probe(), self._manifest())
        available = sorted(
            {
                _normalize_model_name(model)
                for model in probe.models
                if model.strip()
                and _normalize_model_name(model)
                != _normalize_model_name(self.settings.embedding_model)
            }
        )
        installed = {_normalize_model_name(model) for model in probe.models}
        required = {
            _normalize_model_name(self.settings.embedding_model),
            _normalize_model_name(self.settings.llm_model),
        }
        ready = probe.reachable and required.issubset(installed)
        active = self.coordinator.snapshot()
        loaded = self._graph is not None
        idle = active is None
        return RuntimeSnapshot(
            state="ready" if loaded else ("not_loaded" if ready else "blocked"),
            configured_chat_model=self.settings.llm_model,
            active_chat_model=self._active_chat_model,
            embedding_model=self.settings.embedding_model,
            available_chat_models=available,
            detail=(
                "Models are loaded and ready."
                if loaded
                else (
                    "Required local models are available."
                    if ready
                    else "Start Ollama and install the configured local models."
                )
            ),
            capabilities=CapabilitySnapshot(
                can_query=loaded and idle,
                can_load_models=ready and not loaded and idle,
                can_upload=probe.reachable
                and _normalize_model_name(self.settings.embedding_model) in installed
                and idle,
                can_run_benchmark=(self._benchmark_available and loaded and idle),
            ),
            active_operation=active,
            corpus=self._corpus(manifest),
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
    def _latest_evaluation_exists() -> bool:
        results_root = PROJECT_ROOT / "evals" / "results" / "multihop"
        for summary_path in results_root.rglob("summary.json"):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    is_standard_benchmark_summary(summary)
                    and (summary_path.parent / "cases.jsonl").is_file()
                ):
                    return True
            except (OSError, json.JSONDecodeError):
                continue
        return False

    @classmethod
    def _default_evaluation_probe(cls) -> EvaluationProbeResult:
        return EvaluationProbeResult(
            dataset_ready=(PROJECT_ROOT / "evals" / "multihop" / "cases.jsonl").is_file(),
            latest_compatible=cls._latest_evaluation_exists(),
        )

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
        except Exception:
            index_checks = [
                DiagnosticCheck(
                    area="index",
                    name="Index diagnostics",
                    state="error",
                    detail="The local index could not be inspected.",
                )
            ]
        try:
            evaluation = await asyncio.to_thread(self._evaluation_probe)
        except Exception:
            evaluation = EvaluationProbeResult(dataset_ready=False, latest_compatible=False)
        evaluation_checks = [
            DiagnosticCheck(
                area="evaluation",
                name="Benchmark dataset",
                state="ready" if evaluation.dataset_ready else "blocked",
                detail="Benchmark dataset is ready."
                if evaluation.dataset_ready
                else "Prepare the benchmark dataset.",
            ),
            DiagnosticCheck(
                area="evaluation",
                name="Latest compatible evaluation",
                state="ready" if evaluation.latest_compatible else "not_loaded",
                detail="A compatible result is available."
                if evaluation.latest_compatible
                else "No compatible benchmark result is stored.",
            ),
        ]
        all_checks = [*runtime_checks, *index_checks, *evaluation_checks]
        states = {check.state for check in all_checks}
        state = (
            "blocked"
            if "blocked" in states
            else "error"
            if "error" in states
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

    @staticmethod
    def _index_diagnostics(
        manager: _VectorDB,
    ) -> tuple[IngestionManifest, ReconciliationResult, int]:
        return manager.manifest(), manager.reconcile_index(), manager.chunk_count()

    @staticmethod
    def _index_checks(
        manifest: IngestionManifest,
        reconciliation: ReconciliationResult,
        chunks: int,
    ) -> list[DiagnosticCheck]:
        values = [
            ("Missing Chroma chunks", reconciliation.missing_chunk_ids, "blocked"),
            ("Orphan Chroma chunks", reconciliation.orphan_chunk_ids, "review"),
            ("Duplicate IDs", reconciliation.duplicate_chunk_ids, "blocked"),
            ("Missing source files", reconciliation.missing_source_files, "review"),
            ("Index configuration", reconciliation.incompatible_document_ids, "blocked"),
        ]
        checks = [
            DiagnosticCheck(
                area="index",
                name="Chroma collection",
                state="ready",
                detail=f"{chunks} chunks.",
            ),
            DiagnosticCheck(
                area="index",
                name="Manifest",
                state="ready",
                detail=f"Valid; {len(manifest.documents)} documents.",
            ),
        ]
        checks.extend(
            DiagnosticCheck(
                area="index",
                name=name,
                state=cast(Any, failure_state if problems else "ready"),
                detail=f"{len(problems)} found.",
            )
            for name, problems, failure_state in values
        )
        return checks

    async def list_documents(self) -> DocumentList:
        try:
            manifest = await asyncio.to_thread(self._manager().manifest)
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
