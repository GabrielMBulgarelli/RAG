"""Presentation-neutral application exceptions."""

from pydantic import JsonValue

from modules.application.models import ActiveOperation


class ApplicationError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, JsonValue],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class OperationBusyError(ApplicationError):
    def __init__(self, active_operation: ActiveOperation) -> None:
        details: dict[str, JsonValue] = {
            "operation": active_operation.kind.value,
            "operation_id": str(active_operation.operation_id),
        }
        if active_operation.benchmark_run_id is not None:
            details["benchmark_run_id"] = str(active_operation.benchmark_run_id)
        super().__init__(
            code="operation_busy",
            message="Another workspace operation is currently running.",
            details=details,
        )
