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


class InvalidUploadError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="invalid_upload",
            message="One or more uploaded files are invalid.",
            details={},
        )


class UploadLimitExceededApplicationError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="upload_limit_exceeded",
            message="The upload exceeds the workspace limits.",
            details={},
        )


class DocumentNotFoundError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="document_not_found",
            message="The requested document was not found.",
            details={},
        )


class RuntimeUnavailableError(ApplicationError):
    def __init__(self, *, operation: str) -> None:
        super().__init__(
            code="runtime_unavailable",
            message="A required local runtime dependency is unavailable.",
            details={"operation": operation},
        )


class IndexStateError(ApplicationError):
    def __init__(self, *, reason: str) -> None:
        actions = {
            "manifest_invalid": (
                "Restore or remove the invalid manifest, then rebuild the local index."
            ),
            "collection_unavailable": "Rebuild the local index.",
        }
        super().__init__(
            code="index_error",
            message="The persisted index is unavailable.",
            details={"reason": reason, "action": actions[reason]},
        )


class BenchmarkNotFoundError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="benchmark_not_found",
            message="The requested benchmark artifact was not found.",
            details={},
        )


class BenchmarkManagerClosedError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="benchmark_manager_closed",
            message="The benchmark manager has been closed.",
            details={},
        )


class BenchmarkUnavailableError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            code="benchmark_unavailable",
            message="Benchmark execution is not configured.",
            details={},
        )
