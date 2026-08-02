"""Thin HTTP adapters for the workspace API."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Header, Response, UploadFile, status

from modules.api.dependencies import (
    BenchmarkManager,
    UploadedFile,
    WorkspaceService,
    get_benchmarks,
    get_workspace,
)
from modules.application.models import (
    BenchmarkCaseDetail,
    BenchmarkCaseSummary,
    BenchmarkRun,
    BenchmarkStartResponse,
    ConversationExportRequest,
    DiagnosticsSnapshot,
    DocumentList,
    ModelLoadRequest,
    QueryRequest,
    QueryResponse,
    RuntimeSnapshot,
    UploadBatchResult,
)

router = APIRouter(prefix="/api")
WorkspaceDependency = Annotated[WorkspaceService, Depends(get_workspace)]
BenchmarkDependency = Annotated[BenchmarkManager, Depends(get_benchmarks)]


@router.get("/runtime", response_model=RuntimeSnapshot)
async def get_runtime(workspace: WorkspaceDependency) -> RuntimeSnapshot:
    return await workspace.get_runtime()


@router.post("/runtime/models", response_model=RuntimeSnapshot)
async def load_model(
    request: ModelLoadRequest,
    workspace: WorkspaceDependency,
) -> RuntimeSnapshot:
    return await workspace.load_model(request)


@router.get("/diagnostics", response_model=DiagnosticsSnapshot)
async def get_diagnostics(workspace: WorkspaceDependency) -> DiagnosticsSnapshot:
    return await workspace.get_diagnostics()


@router.get("/documents", response_model=DocumentList)
async def list_documents(workspace: WorkspaceDependency) -> DocumentList:
    return await workspace.list_documents()


@router.post("/documents", response_model=UploadBatchResult)
async def upload_documents(
    files: Annotated[list[UploadFile], File()],
    workspace: WorkspaceDependency,
) -> UploadBatchResult:
    uploaded: list[UploadedFile] = []
    for file in files:
        uploaded.append(
            UploadedFile(
                filename=file.filename or "",
                content_type=file.content_type,
                content=await file.read(),
            )
        )
    return await workspace.upload_documents(tuple(uploaded))


@router.delete("/documents/{document_id}", response_model=DocumentList)
async def delete_document(
    document_id: str,
    workspace: WorkspaceDependency,
) -> DocumentList:
    return await workspace.delete_document(document_id)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, workspace: WorkspaceDependency) -> QueryResponse:
    return await workspace.query(request)


@router.delete(
    "/conversations/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def clear_conversation(session_id: UUID, workspace: WorkspaceDependency) -> Response:
    await workspace.clear_conversation(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/conversations/export", response_class=Response)
async def export_conversation(
    request: ConversationExportRequest,
    workspace: WorkspaceDependency,
) -> Response:
    return await workspace.export_conversation(request)


@router.post(
    "/benchmarks",
    response_model=BenchmarkStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_benchmark(benchmarks: BenchmarkDependency) -> BenchmarkStartResponse:
    return await benchmarks.start_benchmark()


@router.get("/benchmarks/latest", response_model=BenchmarkRun)
async def latest_benchmark(benchmarks: BenchmarkDependency) -> BenchmarkRun:
    return await benchmarks.latest_benchmark()


@router.get("/benchmarks/{run_id}", response_model=BenchmarkRun)
async def get_benchmark(run_id: UUID, benchmarks: BenchmarkDependency) -> BenchmarkRun:
    return await benchmarks.get_benchmark(run_id)


@router.get("/benchmarks/{run_id}/cases", response_model=list[BenchmarkCaseSummary])
async def list_benchmark_cases(
    *,
    run_id: UUID,
    benchmarks: BenchmarkDependency,
) -> list[BenchmarkCaseSummary]:
    return await benchmarks.list_cases(run_id)


@router.get(
    "/benchmarks/{run_id}/cases/{case_id}/systems/{system_id}",
    response_model=BenchmarkCaseDetail,
)
async def get_benchmark_case(
    run_id: UUID,
    case_id: str,
    system_id: str,
    benchmarks: BenchmarkDependency,
) -> BenchmarkCaseDetail:
    return await benchmarks.get_case(
        run_id=run_id,
        case_id=case_id,
        system_id=system_id,
    )


@router.get("/benchmarks/{run_id}/events", response_class=Response)
async def stream_benchmark_events(
    run_id: UUID,
    benchmarks: BenchmarkDependency,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID", ge=0)] = None,
) -> Response:
    return await benchmarks.stream_events(
        run_id=run_id,
        last_event_id=last_event_id,
    )


@router.post(
    "/benchmarks/{run_id}/cancel",
    response_model=BenchmarkRun,
    status_code=status.HTTP_202_ACCEPTED,
)
async def cancel_benchmark(run_id: UUID, benchmarks: BenchmarkDependency) -> BenchmarkRun:
    return await benchmarks.cancel_benchmark(run_id)


@router.get("/benchmarks/{run_id}/download", response_class=Response)
async def download_benchmark(run_id: UUID, benchmarks: BenchmarkDependency) -> Response:
    return await benchmarks.download_benchmark(run_id)
