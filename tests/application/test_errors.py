from datetime import datetime, timezone
from uuid import UUID

from modules.application.errors import ApplicationError, OperationBusyError
from modules.application.models import ActiveOperation, OperationKind


def test_application_error_exposes_presentation_neutral_json_details() -> None:
    error = ApplicationError(
        code="invalid_request",
        message="The request is invalid.",
        details={"field": "question", "retryable": False, "attempts": 0},
    )

    assert str(error) == "The request is invalid."
    assert error.code == "invalid_request"
    assert error.message == "The request is invalid."
    assert error.details == {
        "field": "question",
        "retryable": False,
        "attempts": 0,
    }


def test_operation_busy_error_has_exact_sanitized_details() -> None:
    operation_id = UUID("0b5e8a4b-a5df-4a93-a796-4d8cce1f4367")
    run_id = UUID("4cbdbcb9-5a57-4514-a392-2dce907456d5")
    active = ActiveOperation(
        operation_id=operation_id,
        kind=OperationKind.BENCHMARK,
        started_at=datetime(2026, 7, 29, 15, 30, tzinfo=timezone.utc),
        benchmark_run_id=run_id,
    )

    error = OperationBusyError(active)

    assert error.code == "operation_busy"
    assert error.message == "Another workspace operation is currently running."
    assert error.details == {
        "operation": "benchmark",
        "operation_id": str(operation_id),
        "benchmark_run_id": str(run_id),
    }

    without_benchmark = OperationBusyError(
        active.model_copy(
            update={
                "kind": OperationKind.QUERY,
                "benchmark_run_id": None,
            }
        )
    )
    assert without_benchmark.details == {
        "operation": "query",
        "operation_id": str(operation_id),
    }
