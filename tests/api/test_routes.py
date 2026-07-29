from datetime import datetime, timezone
from typing import cast
from uuid import UUID

from fastapi.testclient import TestClient
from starlette.responses import Response, StreamingResponse

from modules.api.app import create_app
from modules.api.dependencies import (
    ApplicationContainer,
    BenchmarkManager,
    UploadedFile,
)
from modules.application.models import (
    BenchmarkCaseDetail,
    BenchmarkMetadata,
    BenchmarkProgress,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkStartResponse,
    BenchmarkSystem,
    CapabilitySnapshot,
    ConversationExportRequest,
    ConversationMessage,
    CorpusSnapshot,
    DiagnosticCheck,
    DiagnosticsSnapshot,
    DocumentList,
    DocumentRecord,
    ModelLoadRequest,
    QueryDiagnostics,
    QueryRequest,
    QueryResponse,
    ResourceLinks,
    RetrievalHit,
    RuntimeSnapshot,
    Source,
    TraceEvent,
    UploadAccepted,
    UploadBatchResult,
)

SESSION_ID = UUID("26bc395b-c7ae-4111-a1e9-bb3677de67f7")
RUN_ID = UUID("4cbdbcb9-5a57-4514-a392-2dce907456d5")
DOCUMENT_ID = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
NOW = datetime(2026, 7, 29, 15, 30, tzinfo=timezone.utc)


def corpus() -> CorpusSnapshot:
    return CorpusSnapshot(document_count=1, page_count=2, chunk_count=3, status="ready")


def runtime() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        state="ready",
        configured_chat_model="qwen3:4b",
        active_chat_model="qwen3:4b",
        embedding_model="nomic-embed-text",
        available_chat_models=["qwen3:4b"],
        detail="Ready.",
        capabilities=CapabilitySnapshot(
            can_query=True,
            can_load_models=True,
            can_upload=True,
            can_run_benchmark=True,
        ),
        active_operation=None,
        corpus=corpus(),
    )


def diagnostics() -> DiagnosticsSnapshot:
    return DiagnosticsSnapshot(
        state="ready",
        title="Ready",
        detail="All checks passed.",
        runtime_checks=[
            DiagnosticCheck(area="runtime", name="Chat model", state="ready", detail="Loaded.")
        ],
        index_checks=[],
        evaluation_checks=[],
        active_operation=None,
        stale=False,
    )


def document_list() -> DocumentList:
    return DocumentList(
        documents=[
            DocumentRecord(
                id=DOCUMENT_ID,
                filename="guide.pdf",
                state="indexed",
                size_bytes=4096,
                page_count=2,
                chunk_count=3,
                indexed_at=NOW,
                updated_at=NOW,
            )
        ],
        corpus=corpus(),
        active_operation=None,
    )


def query_response(request: QueryRequest) -> QueryResponse:
    return QueryResponse(
        session_id=request.session_id,
        message=ConversationMessage(
            id=UUID("b3c19351-5ae1-48e3-938f-c9306359478f"),
            role="assistant",
            content="The limit is 4.",
            created_at=NOW,
        ),
        answer_state="supported",
        sources=[Source(label="C1", filename="guide.pdf", page=2, excerpt="The limit is 4.")],
        retrieval_hits=[
            RetrievalHit(
                chunk_id="chunk-1",
                filename="guide.pdf",
                page=2,
                semantic_score=0.8,
                sparse_score=None,
                fused_score=0.7,
                selection_score=0.9,
                matched_subqueries=["limit"],
            )
        ],
        trace=[
            TraceEvent(
                stage="retrieve",
                decision="hybrid",
                retrieved_count=1,
                fused_count=1,
                selected_count=1,
                retry_count=0,
                llm_calls=0,
                termination="continue",
                duration_ms=1.0,
            )
        ],
        diagnostics=QueryDiagnostics(
            route="simple_search",
            retrieval_strategy="hybrid",
            subqueries=["limit"],
            retry_count=0,
            evidence_state="sufficient",
            conflict_state="none",
            citation_validation="valid",
        ),
    )


class FakeWorkspace:
    def __init__(self) -> None:
        self.uploads: tuple[UploadedFile, ...] = ()
        self.calls: list[tuple[str, object]] = []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def get_runtime(self) -> RuntimeSnapshot:
        self.calls.append(("get_runtime", None))
        return runtime()

    async def load_model(self, request: ModelLoadRequest) -> RuntimeSnapshot:
        self.calls.append(("load_model", request))
        return runtime()

    async def get_diagnostics(self) -> DiagnosticsSnapshot:
        self.calls.append(("get_diagnostics", None))
        return diagnostics()

    async def list_documents(self) -> DocumentList:
        self.calls.append(("list_documents", None))
        return document_list()

    async def upload_documents(self, files: tuple[UploadedFile, ...]) -> UploadBatchResult:
        self.uploads = files
        self.calls.append(("upload_documents", files))
        return UploadBatchResult(
            accepted=[
                UploadAccepted(filename=file.filename, document_id=f"document-{index}")
                for index, file in enumerate(files, start=1)
            ],
            documents=document_list().documents,
            corpus=corpus(),
        )

    async def delete_document(self, document_id: str) -> DocumentList:
        self.calls.append(("delete_document", document_id))
        return document_list()

    async def query(self, request: QueryRequest) -> QueryResponse:
        self.calls.append(("query", request))
        return query_response(request)

    async def clear_conversation(self, session_id: UUID) -> None:
        self.calls.append(("clear_conversation", session_id))

    async def export_conversation(self, request: ConversationExportRequest) -> Response:
        self.calls.append(("export_conversation", request))
        return Response(
            b"# Transcript",
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="conversation.md"'},
        )


class EmptyBenchmarks:
    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


def benchmark_links() -> ResourceLinks:
    return ResourceLinks(
        run=f"/api/benchmarks/{RUN_ID}",
        events=f"/api/benchmarks/{RUN_ID}/events",
        download=f"/api/benchmarks/{RUN_ID}/download",
    )


def benchmark_run(status: BenchmarkRunStatus = BenchmarkRunStatus.RUNNING) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=RUN_ID,
        status=status,
        progress=BenchmarkProgress(completed_cases=1, total_cases=4),
        metadata=BenchmarkMetadata(
            dataset="multihop",
            split="development",
            systems=[BenchmarkSystem(id="dense", label="Dense")],
            chat_model="qwen3:4b",
            embedding_model="nomic-embed-text",
            started_at=NOW,
            completed_at=None,
        ),
        sections=[],
        failures=[],
        links=benchmark_links(),
        error=None,
    )


class FakeBenchmarks:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def start_benchmark(self) -> BenchmarkStartResponse:
        self.calls.append(("start_benchmark", None))
        return BenchmarkStartResponse(
            run_id=RUN_ID,
            status=BenchmarkRunStatus.QUEUED,
            links=benchmark_links(),
        )

    async def latest_benchmark(self) -> BenchmarkRun:
        self.calls.append(("latest_benchmark", None))
        return benchmark_run()

    async def get_benchmark(self, run_id: UUID) -> BenchmarkRun:
        self.calls.append(("get_benchmark", run_id))
        return benchmark_run()

    async def get_case(self, run_id: UUID, case_id: str, system_id: str) -> BenchmarkCaseDetail:
        self.calls.append(("get_case", (run_id, case_id, system_id)))
        return BenchmarkCaseDetail(
            case_id=case_id,
            system=system_id,
            question="What is the limit?",
            expected_answer="4",
            generated_answer="The limit is 4.",
            expected_evidence=[],
            retrieved_evidence=[],
            metric_observations=[],
            failure_classification=None,
            public_trace=[],
            sanitized_raw_result=None,
        )

    async def stream_events(self, run_id: UUID, last_event_id: int | None) -> Response:
        self.calls.append(("stream_events", (run_id, last_event_id)))
        return StreamingResponse(
            iter([b"event: heartbeat\ndata: {}\n\n"]), media_type="text/event-stream"
        )

    async def cancel_benchmark(self, run_id: UUID) -> BenchmarkRun:
        self.calls.append(("cancel_benchmark", run_id))
        return benchmark_run(BenchmarkRunStatus.CANCELLATION_REQUESTED)

    async def download_benchmark(self, run_id: UUID) -> Response:
        self.calls.append(("download_benchmark", run_id))
        return Response(
            b'{"run_id":"download"}',
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="benchmark.json"'},
        )


def test_workspace_endpoints_delegate_and_preserve_http_semantics() -> None:
    workspace = FakeWorkspace()
    app = create_app(
        lambda: ApplicationContainer(
            workspace=workspace,
            benchmarks=cast(BenchmarkManager, EmptyBenchmarks()),
        )
    )

    with TestClient(app) as client:
        assert client.get("/api/runtime").json()["state"] == "ready"
        assert (
            client.post("/api/runtime/models", json={"chat_model": " qwen3:4b "}).json()["state"]
            == "ready"
        )
        assert client.get("/api/diagnostics").json()["state"] == "ready"
        assert client.get("/api/documents").json()["documents"][0]["id"] == DOCUMENT_ID

        upload = client.post(
            "/api/documents",
            files=[
                ("files", ("a.txt", b"alpha", "text/plain")),
                ("files", ("b.pdf", b"beta", "application/pdf")),
            ],
        )
        assert upload.status_code == 200
        assert [item["filename"] for item in upload.json()["accepted"]] == ["a.txt", "b.pdf"]
        assert workspace.uploads == (
            UploadedFile(filename="a.txt", content_type="text/plain", content=b"alpha"),
            UploadedFile(filename="b.pdf", content_type="application/pdf", content=b"beta"),
        )

        assert (
            client.delete(f"/api/documents/{DOCUMENT_ID}").json()["documents"][0]["id"]
            == DOCUMENT_ID
        )
        query = client.post(
            "/api/query",
            json={"session_id": str(SESSION_ID), "question": "What is the limit?"},
        )
        assert query.json()["answer_state"] == "supported"
        cleared = client.delete(f"/api/conversations/{SESSION_ID}")
        assert cleared.status_code == 204
        assert cleared.content == b""
        exported = client.post(
            "/api/conversations/export",
            json={"session_id": str(SESSION_ID)},
        )
        assert exported.content == b"# Transcript"
        assert exported.headers["content-disposition"] == 'attachment; filename="conversation.md"'

        schemas = app.openapi()["paths"]
        assert schemas["/api/runtime"]["get"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"].endswith("/RuntimeSnapshot")
        assert schemas["/api/query"]["post"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"].endswith("/QueryResponse")

    assert [name for name, _ in workspace.calls] == [
        "get_runtime",
        "load_model",
        "get_diagnostics",
        "list_documents",
        "upload_documents",
        "delete_document",
        "query",
        "clear_conversation",
        "export_conversation",
    ]


def test_benchmark_endpoints_delegate_with_static_routing_and_response_semantics() -> None:
    benchmarks = FakeBenchmarks()
    app = create_app(
        lambda: ApplicationContainer(
            workspace=FakeWorkspace(),
            benchmarks=benchmarks,
        )
    )

    with TestClient(app) as client:
        started = client.post("/api/benchmarks")
        assert started.status_code == 202
        assert started.json()["status"] == "queued"

        latest = client.get("/api/benchmarks/latest")
        assert latest.status_code == 200
        assert latest.json()["run_id"] == str(RUN_ID)
        assert client.get(f"/api/benchmarks/{RUN_ID}").json()["status"] == "running"
        case = client.get(f"/api/benchmarks/{RUN_ID}/cases/case-1/systems/dense")
        assert case.json()["case_id"] == "case-1"

        events = client.get(f"/api/benchmarks/{RUN_ID}/events")
        assert events.headers["content-type"].startswith("text/event-stream")
        assert events.content == b"event: heartbeat\ndata: {}\n\n"
        replayed_events = client.get(
            f"/api/benchmarks/{RUN_ID}/events",
            headers={"Last-Event-ID": "0"},
        )
        assert replayed_events.content == b"event: heartbeat\ndata: {}\n\n"

        cancelled = client.post(f"/api/benchmarks/{RUN_ID}/cancel")
        assert cancelled.status_code == 202
        assert cancelled.json()["status"] == "cancellation_requested"

        download = client.get(f"/api/benchmarks/{RUN_ID}/download")
        assert download.content == b'{"run_id":"download"}'
        assert download.headers["content-disposition"] == 'attachment; filename="benchmark.json"'

        schemas = app.openapi()["paths"]
        assert schemas["/api/benchmarks"]["post"]["responses"]["202"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/BenchmarkStartResponse")
        assert schemas["/api/benchmarks/latest"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]["$ref"].endswith("/BenchmarkRun")
        assert schemas["/api/benchmarks/{run_id}/cases/{case_id}/systems/{system_id}"]["get"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/BenchmarkCaseDetail")

    assert [name for name, _ in benchmarks.calls] == [
        "start_benchmark",
        "latest_benchmark",
        "get_benchmark",
        "get_case",
        "stream_events",
        "stream_events",
        "cancel_benchmark",
        "download_benchmark",
    ]
    assert benchmarks.calls[4][1] == (RUN_ID, None)
    assert benchmarks.calls[5][1] == (RUN_ID, 0)
