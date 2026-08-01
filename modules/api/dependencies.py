"""Typed dependencies owned by the FastAPI application lifespan."""

from dataclasses import dataclass
from typing import Annotated, Protocol
from uuid import UUID

from fastapi import Depends, Request
from starlette.responses import Response

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
    UploadedFile,
)


class WorkspaceService(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def get_runtime(self) -> RuntimeSnapshot: ...

    async def load_model(self, request: ModelLoadRequest) -> RuntimeSnapshot: ...

    async def get_diagnostics(self) -> DiagnosticsSnapshot: ...

    async def list_documents(self) -> DocumentList: ...

    async def upload_documents(self, files: tuple[UploadedFile, ...]) -> UploadBatchResult: ...

    async def delete_document(self, document_id: str) -> DocumentList: ...

    async def query(self, request: QueryRequest) -> QueryResponse: ...

    async def clear_conversation(self, session_id: UUID) -> None: ...

    async def export_conversation(self, request: ConversationExportRequest) -> Response: ...


class BenchmarkManager(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def start_benchmark(self) -> BenchmarkStartResponse: ...

    async def latest_benchmark(self) -> BenchmarkRun: ...

    async def get_benchmark(self, run_id: UUID) -> BenchmarkRun: ...

    async def list_cases(self, run_id: UUID) -> list[BenchmarkCaseSummary]: ...

    async def get_case(
        self,
        *,
        run_id: UUID,
        case_id: str,
        system_id: str,
    ) -> BenchmarkCaseDetail: ...

    async def stream_events(
        self,
        *,
        run_id: UUID,
        last_event_id: int | None,
    ) -> Response: ...

    async def cancel_benchmark(self, run_id: UUID) -> BenchmarkRun: ...

    async def download_benchmark(self, run_id: UUID) -> Response: ...


@dataclass(frozen=True)
class ApplicationContainer:
    workspace: WorkspaceService
    benchmarks: BenchmarkManager


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container


def get_workspace(
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> WorkspaceService:
    return container.workspace


def get_benchmarks(
    container: Annotated[ApplicationContainer, Depends(get_container)],
) -> BenchmarkManager:
    return container.benchmarks
