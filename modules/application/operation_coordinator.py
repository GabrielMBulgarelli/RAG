"""Thread-safe coordination for mutually exclusive workspace operations."""

from datetime import datetime, timezone
from threading import Lock
from types import TracebackType
from typing import Self
from uuid import UUID, uuid4

from modules.application.errors import OperationBusyError
from modules.application.models import ActiveOperation, OperationKind


class OperationLease:
    def __init__(
        self,
        coordinator: "WorkspaceOperationCoordinator",
        operation: ActiveOperation,
    ) -> None:
        self._coordinator = coordinator
        self.operation = operation

    def release(self) -> None:
        self._coordinator._release(self.operation.operation_id)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class WorkspaceOperationCoordinator:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active: ActiveOperation | None = None

    def acquire(
        self,
        kind: OperationKind,
        *,
        benchmark_run_id: UUID | None = None,
    ) -> OperationLease:
        operation = ActiveOperation(
            operation_id=uuid4(),
            kind=kind,
            started_at=datetime.now(timezone.utc),
            benchmark_run_id=benchmark_run_id,
        )
        with self._lock:
            if self._active is not None:
                raise OperationBusyError(self._active)
            self._active = operation
        return OperationLease(self, operation)

    def snapshot(self) -> ActiveOperation | None:
        with self._lock:
            return self._active.model_copy(deep=True) if self._active is not None else None

    def request_benchmark_cancellation(self, run_id: UUID) -> ActiveOperation | None:
        with self._lock:
            if (
                self._active is None
                or self._active.kind is not OperationKind.BENCHMARK
                or self._active.benchmark_run_id != run_id
            ):
                return None
            self._active = self._active.model_copy(update={"cancellation_requested": True})
            return self._active.model_copy(deep=True)

    def _release(self, operation_id: UUID) -> None:
        with self._lock:
            if self._active is not None and self._active.operation_id == operation_id:
                self._active = None
