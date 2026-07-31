"""Explicit unavailable adapter used until a production benchmark executor exists."""

from uuid import UUID

from starlette.responses import Response

from modules.application.errors import BenchmarkUnavailableError
from modules.application.models import (
    BenchmarkCaseDetail,
    BenchmarkRun,
    BenchmarkStartResponse,
)


class UnavailableBenchmarkManager:
    """Fulfill the benchmark boundary without pretending execution is supported."""

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def start_benchmark(self) -> BenchmarkStartResponse:
        raise BenchmarkUnavailableError()

    async def latest_benchmark(self) -> BenchmarkRun:
        raise BenchmarkUnavailableError()

    async def get_benchmark(self, run_id: UUID) -> BenchmarkRun:
        raise BenchmarkUnavailableError()

    async def get_case(
        self,
        *,
        run_id: UUID,
        case_id: str,
        system_id: str,
    ) -> BenchmarkCaseDetail:
        raise BenchmarkUnavailableError()

    async def stream_events(
        self,
        *,
        run_id: UUID,
        last_event_id: int | None,
    ) -> Response:
        raise BenchmarkUnavailableError()

    async def cancel_benchmark(self, run_id: UUID) -> BenchmarkRun:
        raise BenchmarkUnavailableError()

    async def download_benchmark(self, run_id: UUID) -> Response:
        raise BenchmarkUnavailableError()
