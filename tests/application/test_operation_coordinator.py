from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from threading import Barrier
from uuid import UUID

import pytest

from modules.application.errors import OperationBusyError
from modules.application.models import OperationKind
from modules.application.operation_coordinator import (
    OperationLease,
    WorkspaceOperationCoordinator,
)


def test_acquire_records_operation_and_snapshot_is_defensive() -> None:
    coordinator = WorkspaceOperationCoordinator()

    lease = coordinator.acquire(OperationKind.QUERY)
    first_snapshot = coordinator.snapshot()
    second_snapshot = coordinator.snapshot()

    assert lease.operation.kind is OperationKind.QUERY
    assert lease.operation.started_at.tzinfo is timezone.utc
    assert first_snapshot == lease.operation
    assert first_snapshot is not lease.operation
    assert second_snapshot == first_snapshot
    assert second_snapshot is not first_snapshot

    lease.release()
    assert coordinator.snapshot() is None


def test_acquire_raises_immediate_busy_error_for_active_operation() -> None:
    coordinator = WorkspaceOperationCoordinator()
    lease = coordinator.acquire(OperationKind.LOAD_MODEL)

    with pytest.raises(OperationBusyError) as raised:
        coordinator.acquire(OperationKind.QUERY)

    assert raised.value.details == {
        "operation": "load_model",
        "operation_id": str(lease.operation.operation_id),
    }
    assert coordinator.snapshot() == lease.operation
    lease.release()


def test_context_manager_releases_after_completion_and_exception() -> None:
    coordinator = WorkspaceOperationCoordinator()

    with coordinator.acquire(OperationKind.QUERY) as lease:
        assert coordinator.snapshot() == lease.operation
    assert coordinator.snapshot() is None

    with pytest.raises(RuntimeError, match="query failed"):
        with coordinator.acquire(OperationKind.QUERY):
            raise RuntimeError("query failed")
    assert coordinator.snapshot() is None


def test_explicit_release_is_idempotent_and_stale_lease_cannot_release_newer_operation() -> None:
    coordinator = WorkspaceOperationCoordinator()
    stale_lease = coordinator.acquire(OperationKind.INDEX_DOCUMENTS)

    stale_lease.release()
    stale_lease.release()
    current_lease = coordinator.acquire(OperationKind.DELETE_DOCUMENT)
    stale_lease.release()

    assert coordinator.snapshot() == current_lease.operation
    current_lease.release()
    current_lease.release()
    assert coordinator.snapshot() is None


def test_benchmark_cancellation_updates_only_matching_active_run_without_releasing() -> None:
    coordinator = WorkspaceOperationCoordinator()
    run_id = UUID("4cbdbcb9-5a57-4514-a392-2dce907456d5")
    other_run_id = UUID("c982da20-5ac7-44a2-bbbc-8f786ccca16f")
    lease = coordinator.acquire(OperationKind.BENCHMARK, benchmark_run_id=run_id)

    assert coordinator.request_benchmark_cancellation(other_run_id) is None
    assert coordinator.snapshot() == lease.operation

    cancelled = coordinator.request_benchmark_cancellation(run_id)
    active = coordinator.snapshot()

    assert cancelled is not None
    assert cancelled.cancellation_requested is True
    assert active == cancelled
    assert active is not cancelled
    lease.release()
    assert coordinator.snapshot() is None


def test_benchmark_cancellation_returns_none_when_idle_or_non_benchmark() -> None:
    coordinator = WorkspaceOperationCoordinator()
    run_id = UUID("4cbdbcb9-5a57-4514-a392-2dce907456d5")

    assert coordinator.request_benchmark_cancellation(run_id) is None
    lease = coordinator.acquire(OperationKind.QUERY)
    assert coordinator.request_benchmark_cancellation(run_id) is None
    assert coordinator.snapshot() == lease.operation
    lease.release()


def test_simultaneous_acquisitions_allow_exactly_one_winner() -> None:
    coordinator = WorkspaceOperationCoordinator()
    contender_count = 8
    barrier = Barrier(contender_count)

    def attempt_acquisition() -> OperationLease | None:
        barrier.wait()
        try:
            return coordinator.acquire(OperationKind.QUERY)
        except OperationBusyError:
            return None

    with ThreadPoolExecutor(max_workers=contender_count) as executor:
        results = list(executor.map(lambda _: attempt_acquisition(), range(contender_count)))

    leases = [result for result in results if result is not None]
    assert len(leases) == 1
    assert coordinator.snapshot() == leases[0].operation
    leases[0].release()
