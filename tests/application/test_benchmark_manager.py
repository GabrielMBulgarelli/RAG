import asyncio
import io
import json
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from modules.application.benchmark_manager import (
    BenchmarkCancellation,
    BenchmarkExecutionResult,
    BenchmarkManager,
    BenchmarkReporter,
)
from modules.application.errors import ApplicationError, BenchmarkNotFoundError
from modules.application.models import (
    BenchmarkCaseDetail,
    BenchmarkEvent,
    BenchmarkEventType,
    BenchmarkMetadata,
    BenchmarkMetric,
    BenchmarkMetricObservation,
    BenchmarkMetricStatus,
    BenchmarkProgress,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkSection,
    BenchmarkSystem,
)
from modules.application.operation_coordinator import WorkspaceOperationCoordinator
from modules.config import Settings


class IdleExecutor:
    def initial_metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            dataset="fixture",
            split="development",
            systems=[BenchmarkSystem(id="system-a", label="System A")],
            chat_model="chat",
            embedding_model="embedding",
            started_at=None,
            completed_at=None,
        )

    async def execute(self, run_id, reporter, cancellation):
        raise AssertionError("constructor must not execute benchmarks")


class CompletingExecutor(IdleExecutor):
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        await self.release.wait()
        await reporter.publish(
            BenchmarkEventType.SYSTEM_STARTED,
            {"system": "system-a"},
            progress=BenchmarkProgress(
                completed_cases=0,
                total_cases=1,
                current_system="system-a",
            ),
        )
        return BenchmarkExecutionResult(
            sections=[
                BenchmarkSection(
                    id="quality",
                    title="Quality",
                    metrics=[],
                )
            ],
            failures=[],
        )


class FailingExecutor(IdleExecutor):
    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        raise RuntimeError("private executor path C:/secret")


class TerminalPublishingExecutor(IdleExecutor):
    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        await reporter.publish(
            BenchmarkEventType.BENCHMARK_COMPLETED,
            {},
            progress=BenchmarkProgress(completed_cases=0, total_cases=0),
        )
        return BenchmarkExecutionResult(sections=[], failures=[])


class MetadataFailureExecutor(IdleExecutor):
    def initial_metadata(self) -> BenchmarkMetadata:
        raise RuntimeError("metadata unavailable")


class CancellationExecutor(IdleExecutor):
    def __init__(self) -> None:
        self.current_case_started = asyncio.Event()
        self.settle_current_case = asyncio.Event()
        self.started_cases: list[str] = []

    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        self.started_cases.append("case-1")
        progress = BenchmarkProgress(
            completed_cases=0,
            total_cases=2,
            current_system="system-a",
            current_case_id="case-1",
        )
        await reporter.publish(
            BenchmarkEventType.CASE_STARTED,
            {"case_id": "case-1", "system": "system-a"},
            progress=progress,
        )
        self.current_case_started.set()
        await self.settle_current_case.wait()
        case = BenchmarkCaseDetail(
            case_id="case-1",
            system="system-a",
            question="Public question",
            expected_answer="Expected",
            generated_answer="Generated",
            expected_evidence=[],
            retrieved_evidence=[],
            metric_observations=[],
            failure_classification=None,
            public_trace=[],
            sanitized_raw_result=None,
        )
        await reporter.publish(
            BenchmarkEventType.CASE_COMPLETED,
            {"case_id": "case-1", "system": "system-a"},
            progress=progress.model_copy(update={"completed_cases": 1}),
            case=case,
        )
        if not cancellation.is_cancelled:
            self.started_cases.append("case-2")
            await reporter.publish(
                BenchmarkEventType.CASE_STARTED,
                {"case_id": "case-2", "system": "system-a"},
                progress=progress.model_copy(update={"current_case_id": "case-2"}),
            )
        return BenchmarkExecutionResult(sections=[], failures=[])


class BurstExecutor(IdleExecutor):
    def __init__(self, count: int) -> None:
        self.count = count
        self.entered = asyncio.Event()
        self.begin = asyncio.Event()
        self.published = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        self.entered.set()
        await self.begin.wait()
        progress = BenchmarkProgress(completed_cases=0, total_cases=self.count)
        for index in range(self.count):
            await reporter.publish(
                BenchmarkEventType.SYSTEM_STARTED,
                {"index": index},
                progress=progress,
            )
        self.published.set()
        await self.release.wait()
        return BenchmarkExecutionResult(sections=[], failures=[])


class HoldingExecutor(IdleExecutor):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        self.entered.set()
        await self.release.wait()
        return BenchmarkExecutionResult(sections=[], failures=[])


class HoldingFailingExecutor(HoldingExecutor):
    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        self.entered.set()
        await self.release.wait()
        raise RuntimeError("private executor failure")


class CooperativeCloseExecutor(IdleExecutor):
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        self.entered.set()
        await cancellation.wait()
        return BenchmarkExecutionResult(sections=[], failures=[])


class CaseOrderingExecutor(IdleExecutor):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.publish_first = asyncio.Event()
        self.publish_second = asyncio.Event()

    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        self.entered.set()
        await self.publish_first.wait()
        progress = BenchmarkProgress(completed_cases=1, total_cases=1)
        for answer, wait_for_second in (("First", True), ("Revised", False)):
            case = BenchmarkCaseDetail(
                case_id="case-1",
                system="system-a",
                question="Public question",
                expected_answer="Expected",
                generated_answer=answer,
                expected_evidence=[],
                retrieved_evidence=[],
                metric_observations=[],
                failure_classification=None,
                public_trace=[],
                sanitized_raw_result=None,
            )
            await reporter.publish(
                BenchmarkEventType.CASE_COMPLETED,
                {"case_id": "case-1", "system": "system-a"},
                progress=progress,
                case=case,
            )
            if wait_for_second:
                await self.publish_second.wait()
        return BenchmarkExecutionResult(sections=[], failures=[])


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        sources_dir=tmp_path / "sources",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        manifest_path=tmp_path / "data" / "manifest.json",
        trace_dir=tmp_path / "data" / "traces",
        logs_dir=tmp_path / "logs",
        benchmark_results_dir=tmp_path / "benchmark-results",
    )


def test_constructor_does_not_create_storage_or_tasks(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    BenchmarkManager(
        settings=settings,
        coordinator=WorkspaceOperationCoordinator(),
        executor=IdleExecutor(),
    )

    assert not settings.benchmark_results_dir.exists()
    with asyncio.Runner() as runner:
        assert runner.run(asyncio.sleep(0, result=True)) is True


async def wait_for_terminal(manager: BenchmarkManager, run_id: UUID) -> BenchmarkRun:
    for _ in range(200):
        run = await manager.get_benchmark(run_id)
        if run.status in {
            BenchmarkRunStatus.COMPLETED,
            BenchmarkRunStatus.CANCELLED,
            BenchmarkRunStatus.FAILED,
        }:
            return run
        await asyncio.sleep(0.005)
    raise AssertionError("benchmark did not reach a terminal state")


async def response_text(response) -> str:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    return b"".join(chunks).decode("utf-8")


def sse_events(payload: str) -> list[dict]:
    return [
        json.loads(
            next(
                line.removeprefix("data: ")
                for line in block.splitlines()
                if line.startswith("data: ")
            )
        )
        for block in payload.strip().split("\n\n")
        if block.strip()
    ]


def test_queued_run_completes_with_persisted_ordered_journal_and_exact_links(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        executor = CompletingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        await manager.start()

        response = await manager.start_benchmark()

        assert response.status is BenchmarkRunStatus.QUEUED
        assert response.links.model_dump() == {
            "run": f"/api/benchmarks/{response.run_id}",
            "events": f"/api/benchmarks/{response.run_id}/events",
            "download": f"/api/benchmarks/{response.run_id}/download",
        }
        active = coordinator.snapshot()
        assert active is not None
        assert active.benchmark_run_id == response.run_id
        run_dir = settings.benchmark_results_dir / str(response.run_id)
        persisted_queued = BenchmarkRun.model_validate_json(
            (run_dir / "run.json").read_text(encoding="utf-8")
        )
        assert persisted_queued.status is BenchmarkRunStatus.QUEUED
        assert (run_dir / "events.jsonl").read_text(encoding="utf-8") == ""
        assert (run_dir / "cases.jsonl").read_text(encoding="utf-8") == ""

        executor.release.set()
        completed = await wait_for_terminal(manager, response.run_id)

        assert completed.status is BenchmarkRunStatus.COMPLETED
        assert completed.metadata.started_at is not None
        assert completed.metadata.completed_at is not None
        assert [section.id for section in completed.sections] == ["quality"]
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [event["event_id"] for event in events] == [1, 2, 3]
        assert [event["type"] for event in events] == [
            "benchmark.started",
            "system.started",
            "benchmark.completed",
        ]
        assert coordinator.snapshot() is None

        await manager.close()
        await manager.close()

    asyncio.run(scenario())


def test_executor_failure_is_sanitized_and_releases_lease(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=FailingExecutor(),
        )
        await manager.start()
        started = await manager.start_benchmark()

        failed = await wait_for_terminal(manager, started.run_id)

        assert failed.status is BenchmarkRunStatus.FAILED
        assert failed.error is not None
        assert failed.error.code == "benchmark_failed"
        assert "secret" not in failed.model_dump_json()
        events = [
            json.loads(line)
            for line in (settings.benchmark_results_dir / str(started.run_id) / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [event["type"] for event in events] == [
            "benchmark.started",
            "benchmark.failed",
        ]
        assert coordinator.snapshot() is None

    asyncio.run(scenario())


def test_executor_cannot_publish_manager_owned_lifecycle_events(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=TerminalPublishingExecutor(),
        )
        await manager.start()
        started = await manager.start_benchmark()

        failed = await wait_for_terminal(manager, started.run_id)

        assert failed.status is BenchmarkRunStatus.FAILED
        assert failed.error is not None
        assert failed.error.code == "benchmark_failed"
        event_types = [
            json.loads(line)["type"]
            for line in (settings.benchmark_results_dir / str(started.run_id) / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert event_types == ["benchmark.started", "benchmark.failed"]

    asyncio.run(scenario())


def test_start_preparation_or_task_creation_failure_never_leaks_lease_or_active_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        metadata_failure = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=MetadataFailureExecutor(),
        )
        await metadata_failure.start()

        with pytest.raises(RuntimeError):
            await metadata_failure.start_benchmark()
        assert coordinator.snapshot() is None
        assert metadata_failure._active is None

        path_failure = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=IdleExecutor(),
        )
        await path_failure.start()

        def fail_run_dir(run_id: UUID, *, must_exist: bool = False) -> Path:
            raise BenchmarkNotFoundError()

        monkeypatch.setattr(path_failure, "_run_dir", fail_run_dir)
        with pytest.raises(BenchmarkNotFoundError):
            await path_failure.start_benchmark()
        assert coordinator.snapshot() is None
        assert path_failure._active is None

        task_failure = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=IdleExecutor(),
        )
        await task_failure.start()

        def fail_create_task(coroutine, *args, **kwargs):
            coroutine.close()
            raise RuntimeError("task creation unavailable")

        monkeypatch.setattr(asyncio, "create_task", fail_create_task)
        with pytest.raises(RuntimeError):
            await task_failure.start_benchmark()

        assert coordinator.snapshot() is None
        assert task_failure._active is None
        run_dirs = [path for path in settings.benchmark_results_dir.iterdir() if path.is_dir()]
        assert run_dirs == []

    asyncio.run(scenario())


def test_task_creation_and_cleanup_failure_leaves_inspectable_failed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=IdleExecutor(),
        )
        await manager.start()

        def fail_create_task(coroutine, *args, **kwargs):
            coroutine.close()
            raise RuntimeError("task creation unavailable")

        def fail_remove_tree(path: Path) -> None:
            raise OSError("cleanup unavailable")

        monkeypatch.setattr(asyncio, "create_task", fail_create_task)
        monkeypatch.setattr(
            "modules.application.benchmark_manager.shutil.rmtree",
            fail_remove_tree,
        )
        with pytest.raises(RuntimeError):
            await manager.start_benchmark()

        assert coordinator.snapshot() is None
        assert manager._active is None
        run_dirs = [path for path in settings.benchmark_results_dir.iterdir() if path.is_dir()]
        assert len(run_dirs) == 1
        run_id = UUID(run_dirs[0].name)
        failed = await manager.get_benchmark(run_id)
        events = [
            BenchmarkEvent.model_validate_json(line)
            for line in (run_dirs[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert failed.status is BenchmarkRunStatus.FAILED
        assert failed.error is not None
        assert failed.error.code == "benchmark_start_failed"
        assert [event.type for event in events] == [BenchmarkEventType.BENCHMARK_FAILED]

    asyncio.run(scenario())


def test_cancellation_is_idempotent_and_current_case_settles_without_next_case(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        executor = CancellationExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=executor,
        )
        await manager.start()
        started = await manager.start_benchmark()
        await executor.current_case_started.wait()

        first = await manager.cancel_benchmark(started.run_id)
        second = await manager.cancel_benchmark(started.run_id)

        assert first.status is BenchmarkRunStatus.CANCELLATION_REQUESTED
        assert second.status is BenchmarkRunStatus.CANCELLATION_REQUESTED
        active = coordinator.snapshot()
        assert active is not None
        assert active.cancellation_requested is True

        executor.settle_current_case.set()
        cancelled = await wait_for_terminal(manager, started.run_id)

        assert cancelled.status is BenchmarkRunStatus.CANCELLED
        assert executor.started_cases == ["case-1"]
        case = await manager.get_case(started.run_id, "case-1", "system-a")
        assert case.generated_answer == "Generated"
        event_types = [
            json.loads(line)["type"]
            for line in (settings.benchmark_results_dir / str(started.run_id) / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert event_types == [
            "benchmark.started",
            "case.started",
            "benchmark.cancellation_requested",
            "case.completed",
            "benchmark.cancelled",
        ]

    asyncio.run(scenario())


def test_queued_cancellation_never_regresses_to_running(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = CooperativeCloseExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
        )
        await manager.start()
        started = await manager.start_benchmark()

        requested = await manager.cancel_benchmark(started.run_id)
        cancelled = await wait_for_terminal(manager, started.run_id)

        assert requested.status is BenchmarkRunStatus.CANCELLATION_REQUESTED
        assert cancelled.status is BenchmarkRunStatus.CANCELLED
        events = [
            json.loads(line)["type"]
            for line in (settings.benchmark_results_dir / str(started.run_id) / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert events == [
            "benchmark.cancellation_requested",
            "benchmark.cancelled",
        ]

    asyncio.run(scenario())


def test_case_and_run_snapshots_are_durable_before_broadcast_and_last_case_wins(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = CaseOrderingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        stream = await manager.stream_events(started.run_id, None)
        iterator = stream.body_iterator.__aiter__()

        executor.publish_first.set()
        first_chunk = await anext(iterator)
        first_payload = first_chunk if isinstance(first_chunk, str) else bytes(first_chunk).decode()
        assert sse_events(first_payload)[0]["type"] == "case.completed"

        run_dir = settings.benchmark_results_dir / str(started.run_id)
        durable_run = BenchmarkRun.model_validate_json(
            (run_dir / "run.json").read_text(encoding="utf-8")
        )
        durable_cases = [
            BenchmarkCaseDetail.model_validate_json(line)
            for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        durable_events = [
            json.loads(line)["type"]
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert durable_run.progress.completed_cases == 1
        assert [case.generated_answer for case in durable_cases] == ["First"]
        assert durable_events == ["benchmark.started", "case.completed"]

        executor.publish_second.set()
        completed = await wait_for_terminal(manager, started.run_id)
        assert completed.status is BenchmarkRunStatus.COMPLETED
        latest = await manager.get_case(started.run_id, "case-1", "system-a")
        assert latest.generated_answer == "Revised"

        await response_text(stream)
        await manager.close()

    asyncio.run(scenario())


def persisted_run(
    settings: Settings,
    *,
    status: BenchmarkRunStatus,
    completed_at: datetime | None = None,
    sections: list[BenchmarkSection] | None = None,
) -> BenchmarkRun:
    run_id = uuid4()
    run = BenchmarkRun(
        run_id=run_id,
        status=status,
        progress=BenchmarkProgress(completed_cases=0, total_cases=1),
        metadata=BenchmarkMetadata(
            dataset="fixture",
            split="development",
            systems=[BenchmarkSystem(id="system-a", label="System A")],
            chat_model="chat",
            embedding_model="embedding",
            started_at=datetime.now(timezone.utc),
            completed_at=completed_at,
            reproducibility={"seed": 7},
        ),
        sections=sections or [],
        failures=[],
        links=BenchmarkManager._links(run_id),
        error=None,
    )
    run_dir = settings.benchmark_results_dir / str(run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")
    terminal_type = {
        BenchmarkRunStatus.CANCELLED: BenchmarkEventType.BENCHMARK_CANCELLED,
        BenchmarkRunStatus.COMPLETED: BenchmarkEventType.BENCHMARK_COMPLETED,
        BenchmarkRunStatus.FAILED: BenchmarkEventType.BENCHMARK_FAILED,
    }.get(status)
    terminal_event = (
        BenchmarkEvent(
            event_id=1,
            run_id=run_id,
            type=terminal_type,
            timestamp=completed_at or datetime.now(timezone.utc),
            data={},
        ).model_dump_json()
        + "\n"
        if terminal_type is not None
        else ""
    )
    (run_dir / "events.jsonl").write_text(terminal_event, encoding="utf-8")
    (run_dir / "cases.jsonl").write_text("", encoding="utf-8")
    return run


def test_start_recovers_stale_runs_and_latest_uses_completed_timestamp(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        settings.benchmark_results_dir.mkdir(parents=True)
        stale = [
            persisted_run(settings, status=status)
            for status in (
                BenchmarkRunStatus.QUEUED,
                BenchmarkRunStatus.RUNNING,
                BenchmarkRunStatus.CANCELLATION_REQUESTED,
            )
        ]
        missing_journal = persisted_run(
            settings,
            status=BenchmarkRunStatus.RUNNING,
        )
        (settings.benchmark_results_dir / str(missing_journal.run_id) / "events.jsonl").unlink()
        corrupt_journal = persisted_run(
            settings,
            status=BenchmarkRunStatus.CANCELLATION_REQUESTED,
        )
        (settings.benchmark_results_dir / str(corrupt_journal.run_id) / "events.jsonl").write_text(
            "{broken", encoding="utf-8"
        )
        stale.extend((missing_journal, corrupt_journal))
        older = persisted_run(
            settings,
            status=BenchmarkRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc) - timedelta(days=2),
        )
        zero_metric = BenchmarkSection(
            id="quality",
            title="Quality",
            system_ids=["system-a"],
            metrics=[
                BenchmarkMetric(
                    name="score",
                    label="Score",
                    observations=[
                        BenchmarkMetricObservation(
                            system="system-a",
                            value=0.0,
                            status=BenchmarkMetricStatus.MEASURED,
                            sample_count=1,
                        ),
                        BenchmarkMetricObservation(
                            system="system-b",
                            value=None,
                            status=BenchmarkMetricStatus.NOT_APPLICABLE,
                            sample_count=0,
                        ),
                    ],
                )
            ],
        )
        newer = persisted_run(
            settings,
            status=BenchmarkRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc) - timedelta(days=1),
            sections=[zero_metric],
        )
        persisted_run(
            settings,
            status=BenchmarkRunStatus.CANCELLED,
            completed_at=datetime.now(timezone.utc),
        )
        corrupt = settings.benchmark_results_dir / str(uuid4())
        corrupt.mkdir()
        (corrupt / "run.json").write_text("{broken", encoding="utf-8")
        invalid_companion = persisted_run(
            settings,
            status=BenchmarkRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        (
            settings.benchmark_results_dir / str(invalid_companion.run_id) / "events.jsonl"
        ).write_text("{broken", encoding="utf-8")
        unrelated = settings.benchmark_results_dir / "not-a-run"
        unrelated.mkdir()
        (unrelated / "secret.txt").write_text("ignore", encoding="utf-8")
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=IdleExecutor(),
        )

        await manager.start()
        await manager.start()

        for original in stale:
            recovered = await manager.get_benchmark(original.run_id)
            assert recovered.status is BenchmarkRunStatus.FAILED
            assert recovered.error is not None
            assert recovered.error.code == "benchmark_interrupted"
            event_lines = (
                (settings.benchmark_results_dir / str(original.run_id) / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            assert len(event_lines) == 1
            assert json.loads(event_lines[0])["type"] == "benchmark.failed"
        latest = await manager.latest_benchmark()
        assert latest.run_id == newer.run_id
        assert latest.run_id != older.run_id
        observations = latest.sections[0].metrics[0].observations
        assert observations[0].value == 0.0
        assert observations[0].status is BenchmarkMetricStatus.MEASURED
        assert observations[1].value is None
        assert observations[1].status is BenchmarkMetricStatus.NOT_APPLICABLE
        assert latest.metadata.reproducibility == {"seed": 7}

        with pytest.raises(BenchmarkNotFoundError):
            await manager.get_benchmark(invalid_companion.run_id)
        with pytest.raises(BenchmarkNotFoundError):
            await manager.get_benchmark(uuid4())
        with pytest.raises(BenchmarkNotFoundError):
            await manager.get_case(newer.run_id, "missing", "system-a")

        empty_settings = make_settings(tmp_path / "empty")
        empty = BenchmarkManager(
            settings=empty_settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=IdleExecutor(),
        )
        await empty.start()
        with pytest.raises(BenchmarkNotFoundError):
            await empty.latest_benchmark()

    asyncio.run(scenario())


def test_recovery_canonicalizes_terminal_crash_windows(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        settings.benchmark_results_dir.mkdir(parents=True)
        completed_missing_terminal = persisted_run(
            settings,
            status=BenchmarkRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )
        (
            settings.benchmark_results_dir / str(completed_missing_terminal.run_id) / "events.jsonl"
        ).write_text("", encoding="utf-8")

        stale_with_terminal = persisted_run(
            settings,
            status=BenchmarkRunStatus.RUNNING,
        )
        stale_dir = settings.benchmark_results_dir / str(stale_with_terminal.run_id)
        (stale_dir / "events.jsonl").write_text(
            BenchmarkEvent(
                event_id=1,
                run_id=stale_with_terminal.run_id,
                type=BenchmarkEventType.BENCHMARK_COMPLETED,
                timestamp=datetime.now(timezone.utc),
                data={},
            ).model_dump_json()
            + "\n",
            encoding="utf-8",
        )

        completed_mismatched_terminals = persisted_run(
            settings,
            status=BenchmarkRunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )
        mismatched_dir = settings.benchmark_results_dir / str(completed_mismatched_terminals.run_id)
        (mismatched_dir / "events.jsonl").write_text(
            "\n".join(
                event.model_dump_json()
                for event in (
                    BenchmarkEvent(
                        event_id=1,
                        run_id=completed_mismatched_terminals.run_id,
                        type=BenchmarkEventType.BENCHMARK_FAILED,
                        timestamp=datetime.now(timezone.utc),
                        data={},
                    ),
                    BenchmarkEvent(
                        event_id=2,
                        run_id=completed_mismatched_terminals.run_id,
                        type=BenchmarkEventType.BENCHMARK_CANCELLED,
                        timestamp=datetime.now(timezone.utc),
                        data={},
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=IdleExecutor(),
        )
        await manager.start()

        expected = (
            (
                completed_missing_terminal.run_id,
                BenchmarkRunStatus.COMPLETED,
                BenchmarkEventType.BENCHMARK_COMPLETED,
            ),
            (
                stale_with_terminal.run_id,
                BenchmarkRunStatus.FAILED,
                BenchmarkEventType.BENCHMARK_FAILED,
            ),
            (
                completed_mismatched_terminals.run_id,
                BenchmarkRunStatus.COMPLETED,
                BenchmarkEventType.BENCHMARK_COMPLETED,
            ),
        )
        for run_id, expected_status, expected_terminal in expected:
            run = await manager.get_benchmark(run_id)
            events = [
                BenchmarkEvent.model_validate_json(line)
                for line in (settings.benchmark_results_dir / str(run_id) / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            terminals = [
                event
                for event in events
                if event.type
                in {
                    BenchmarkEventType.BENCHMARK_CANCELLED,
                    BenchmarkEventType.BENCHMARK_COMPLETED,
                    BenchmarkEventType.BENCHMARK_FAILED,
                }
            ]
            assert run.status is expected_status
            assert [event.event_id for event in events] == list(range(1, len(events) + 1))
            assert [event.type for event in terminals] == [expected_terminal]
            assert events[-1].type is expected_terminal

    asyncio.run(scenario())


def test_symlinked_run_directories_and_artifacts_are_not_exposed(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    settings.benchmark_results_dir.mkdir(parents=True)
    external_settings = make_settings(tmp_path / "external")
    external_settings.benchmark_results_dir.mkdir(parents=True)
    external_run = persisted_run(
        external_settings,
        status=BenchmarkRunStatus.COMPLETED,
        completed_at=datetime.now(timezone.utc),
    )
    run_link = settings.benchmark_results_dir / str(external_run.run_id)
    local_run = persisted_run(
        settings,
        status=BenchmarkRunStatus.COMPLETED,
        completed_at=datetime.now(timezone.utc),
    )
    local_cases = settings.benchmark_results_dir / str(local_run.run_id) / "cases.jsonl"
    external_cases = tmp_path / "external-cases.jsonl"
    external_cases.write_text(
        BenchmarkCaseDetail(
            case_id="case-1",
            system="system-a",
            question="Public question",
            expected_answer=None,
            generated_answer="External",
            expected_evidence=[],
            retrieved_evidence=[],
            metric_observations=[],
            failure_classification=None,
            public_trace=[],
            sanitized_raw_result=None,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    try:
        run_link.symlink_to(
            external_settings.benchmark_results_dir / str(external_run.run_id),
            target_is_directory=True,
        )
        local_cases.unlink()
        local_cases.symlink_to(external_cases)
    except OSError as exc:
        pytest.skip(f"symlink creation denied by this OS: {exc}")

    async def scenario() -> None:
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=IdleExecutor(),
        )
        await manager.start()

        with pytest.raises(BenchmarkNotFoundError):
            await manager.get_benchmark(external_run.run_id)
        with pytest.raises(BenchmarkNotFoundError):
            await manager.get_benchmark(local_run.run_id)
        with pytest.raises(BenchmarkNotFoundError):
            await manager.get_case(local_run.run_id, "case-1", "system-a")

    asyncio.run(scenario())


def test_terminal_sse_replays_requested_events_and_then_closes(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = CompletingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        started = await manager.start_benchmark()
        executor.release.set()
        await wait_for_terminal(manager, started.run_id)

        future_only = await response_text(await manager.stream_events(started.run_id, None))
        all_events = sse_events(await response_text(await manager.stream_events(started.run_id, 0)))
        after_one = sse_events(await response_text(await manager.stream_events(started.run_id, 1)))

        assert future_only == ""
        assert [event["event_id"] for event in all_events] == [1, 2, 3]
        assert [event["event_id"] for event in after_one] == [2, 3]
        assert all(event["run_id"] == str(started.run_id) for event in all_events)
        rendered = await response_text(await manager.stream_events(started.run_id, 2))
        assert "id: 3\n" in rendered
        assert "event: benchmark.completed\n" in rendered
        assert '"type":"benchmark.completed"' in rendered

    asyncio.run(scenario())


def test_active_replay_uses_disk_before_memory_window_without_gap_or_duplicate(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = BurstExecutor(520)
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        executor.begin.set()
        await executor.published.wait()
        active = manager._active
        assert active is not None
        assert len(active.events) == 512

        replay_response = await manager.stream_events(started.run_id, 0)
        future_cursor_response = await manager.stream_events(started.run_id, 10_000)
        assert {queue.maxsize for queue in active.subscribers} == {64}

        executor.release.set()
        await wait_for_terminal(manager, started.run_id)
        replay = sse_events(await response_text(replay_response))
        future = await response_text(future_cursor_response)

        assert [event["event_id"] for event in replay] == list(range(1, 523))
        assert len({event["event_id"] for event in replay}) == 522
        assert future == ""

    asyncio.run(scenario())


def test_slow_subscriber_overflow_never_blocks_producer_and_closes_stream(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = BurstExecutor(70)
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        response = await manager.stream_events(started.run_id, None)
        active = manager._active
        assert active is not None
        assert len(active.subscribers) == 1
        assert next(iter(active.subscribers)).maxsize == 64

        executor.begin.set()
        await executor.published.wait()
        executor.release.set()
        completed = await wait_for_terminal(manager, started.run_id)
        payload = await asyncio.wait_for(response_text(response), timeout=1)

        assert completed.status is BenchmarkRunStatus.COMPLETED
        assert payload == ""
        assert active.subscribers == set()

    asyncio.run(scenario())


def test_heartbeats_have_monotonic_ids_and_stop_before_terminal(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = HoldingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
            heartbeat_interval=0.005,
        )
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        journal = settings.benchmark_results_dir / str(started.run_id) / "events.jsonl"
        for _ in range(100):
            if len(journal.read_text(encoding="utf-8").splitlines()) >= 3:
                break
            await asyncio.sleep(0.005)
        executor.release.set()
        await wait_for_terminal(manager, started.run_id)
        events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]

        assert [event["event_id"] for event in events] == list(range(1, len(events) + 1))
        assert sum(event["type"] == "heartbeat" for event in events) >= 2
        assert events[-1]["type"] == "benchmark.completed"
        assert all(event["type"] != "heartbeat" for event in events[-1:])

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_point", ["run_write", "event_append"])
def test_heartbeat_persistence_failure_still_writes_one_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        executor = HoldingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=executor,
            heartbeat_interval=0.005,
        )
        original_write_run = manager._write_run
        original_append_event = manager._append_event
        write_count = 0
        heartbeat_append_failed = False

        def fail_one_run_write(run: BenchmarkRun) -> None:
            nonlocal write_count
            write_count += 1
            if failure_point == "run_write" and write_count == 3:
                raise OSError("injected heartbeat run write failure")
            original_write_run(run)

        def fail_one_event_append(event) -> None:
            nonlocal heartbeat_append_failed
            original_append_event(event)
            if (
                failure_point == "event_append"
                and event.type is BenchmarkEventType.HEARTBEAT
                and not heartbeat_append_failed
            ):
                heartbeat_append_failed = True
                raise OSError("injected heartbeat append failure after durable write")

        monkeypatch.setattr(manager, "_write_run", fail_one_run_write)
        monkeypatch.setattr(manager, "_append_event", fail_one_event_append)
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        active = manager._active
        assert active is not None
        for _ in range(200):
            if active.heartbeat_task is not None and active.heartbeat_task.done():
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("heartbeat persistence failure was not observed")

        if failure_point == "event_append":
            requested = await manager.cancel_benchmark(started.run_id)
            assert requested.status is BenchmarkRunStatus.CANCELLATION_REQUESTED
        executor.release.set()
        terminal = await wait_for_terminal(manager, started.run_id)

        expected_status = (
            BenchmarkRunStatus.CANCELLED
            if failure_point == "event_append"
            else BenchmarkRunStatus.FAILED
        )
        assert terminal.status is expected_status
        assert coordinator.snapshot() is None
        events = [
            BenchmarkEvent.model_validate_json(line)
            for line in (settings.benchmark_results_dir / str(started.run_id) / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [event.event_id for event in events] == list(range(1, len(events) + 1))
        terminal_events = [
            event
            for event in events
            if event.type.value.startswith("benchmark.")
            and event.type
            in {
                BenchmarkEventType.BENCHMARK_CANCELLED,
                BenchmarkEventType.BENCHMARK_COMPLETED,
                BenchmarkEventType.BENCHMARK_FAILED,
            }
        ]
        assert len(terminal_events) == 1
        assert terminal_events[0].type is (
            BenchmarkEventType.BENCHMARK_CANCELLED
            if failure_point == "event_append"
            else BenchmarkEventType.BENCHMARK_FAILED
        )

    asyncio.run(scenario())


def test_partial_heartbeat_event_append_restores_journal_before_terminal_and_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        executor = HoldingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=executor,
            heartbeat_interval=0.005,
        )
        original_append_event = manager._append_event
        partial_written = False

        def partially_append_heartbeat(event: BenchmarkEvent) -> None:
            nonlocal partial_written
            if event.type is BenchmarkEventType.HEARTBEAT and not partial_written:
                partial_written = True
                path = manager._artifact_path(event.run_id, "events.jsonl")
                with path.open("a", encoding="utf-8") as stream:
                    stream.write('{"event_id":')
                    stream.flush()
                raise OSError("injected partial heartbeat append")
            original_append_event(event)

        monkeypatch.setattr(manager, "_append_event", partially_append_heartbeat)
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        active = manager._active
        assert active is not None
        assert active.task is not None
        execution_task = active.task
        for _ in range(200):
            if active.heartbeat_task is not None and active.heartbeat_task.done():
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("partial heartbeat append failure was not observed")
        executor.release.set()
        await execution_task

        failed = await manager.get_benchmark(started.run_id)
        events = manager._read_events(started.run_id)
        terminals = [
            event
            for event in events
            if event.type
            in {
                BenchmarkEventType.BENCHMARK_CANCELLED,
                BenchmarkEventType.BENCHMARK_COMPLETED,
                BenchmarkEventType.BENCHMARK_FAILED,
            }
        ]
        assert failed.status is BenchmarkRunStatus.FAILED
        assert failed.error is not None
        assert failed.error.code == "benchmark_failed"
        assert [event.event_id for event in events] == list(range(1, len(events) + 1))
        assert [event.type for event in terminals] == [BenchmarkEventType.BENCHMARK_FAILED]
        assert coordinator.snapshot() is None

        restarted = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=IdleExecutor(),
        )
        await restarted.start()
        reloaded = await restarted.get_benchmark(started.run_id)
        assert reloaded.status is BenchmarkRunStatus.FAILED
        assert [event.event_id for event in restarted._read_events(started.run_id)] == [
            event.event_id for event in events
        ]

    asyncio.run(scenario())


def test_executor_and_heartbeat_failure_still_terminalize_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        executor = HoldingFailingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=executor,
            heartbeat_interval=0.005,
        )
        original_write_run = manager._write_run
        write_count = 0

        def fail_one_heartbeat_write(run: BenchmarkRun) -> None:
            nonlocal write_count
            write_count += 1
            if write_count == 3:
                raise OSError("injected heartbeat run write failure")
            original_write_run(run)

        monkeypatch.setattr(manager, "_write_run", fail_one_heartbeat_write)
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        active = manager._active
        assert active is not None
        for _ in range(200):
            if active.heartbeat_task is not None and active.heartbeat_task.done():
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("heartbeat persistence failure was not observed")
        executor.release.set()

        failed = await wait_for_terminal(manager, started.run_id)

        assert failed.status is BenchmarkRunStatus.FAILED
        assert failed.error is not None
        assert failed.error.code == "benchmark_failed"
        assert coordinator.snapshot() is None
        events = manager._read_events(started.run_id)
        assert [event.type for event in events] == [
            BenchmarkEventType.BENCHMARK_STARTED,
            BenchmarkEventType.BENCHMARK_FAILED,
        ]

    asyncio.run(scenario())


def test_close_requests_cancellation_awaits_completion_and_closes_subscriber(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = CooperativeCloseExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        response = await manager.stream_events(started.run_id, None)

        await manager.close()
        await manager.close()
        payload = sse_events(await asyncio.wait_for(response_text(response), timeout=1))
        cancelled = await manager.get_benchmark(started.run_id)

        assert cancelled.status is BenchmarkRunStatus.CANCELLED
        assert [event["type"] for event in payload] == [
            "benchmark.cancellation_requested",
            "benchmark.cancelled",
        ]
        assert manager._active is None

    asyncio.run(scenario())


def test_start_after_close_is_rejected_without_acquiring_a_lease(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=IdleExecutor(),
        )
        await manager.start()
        await manager.close()

        with pytest.raises(ApplicationError) as raised:
            await manager.start_benchmark()

        assert raised.value.code == "benchmark_manager_closed"
        assert coordinator.snapshot() is None
        assert manager._active is None
        assert list(settings.benchmark_results_dir.iterdir()) == []

    asyncio.run(scenario())


def test_close_waits_for_start_preparation_then_cancels_the_created_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        coordinator = WorkspaceOperationCoordinator()
        executor = CooperativeCloseExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=coordinator,
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        original_write_run = manager._write_run
        preparation_entered = threading.Event()
        release_preparation = threading.Event()
        blocked = False

        def block_queued_write(run: BenchmarkRun) -> None:
            nonlocal blocked
            if run.status is BenchmarkRunStatus.QUEUED and not blocked:
                blocked = True
                preparation_entered.set()
                if not release_preparation.wait(timeout=5):
                    raise TimeoutError("test did not release start preparation")
            original_write_run(run)

        monkeypatch.setattr(manager, "_write_run", block_queued_write)
        start_task = asyncio.create_task(manager.start_benchmark())
        assert await asyncio.to_thread(preparation_entered.wait, 1)
        close_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        close_finished_during_preparation = close_task.done()
        release_preparation.set()
        started = await start_task
        await close_task

        if close_finished_during_preparation:
            await manager.cancel_benchmark(started.run_id)
            active = manager._active
            if active is not None and active.task is not None:
                await active.task

        terminal = await manager.get_benchmark(started.run_id)
        assert close_finished_during_preparation is False
        assert terminal.status is BenchmarkRunStatus.CANCELLED
        assert coordinator.snapshot() is None
        assert manager._active is None

    asyncio.run(scenario())


def test_stream_disconnect_removes_subscriber_ownership(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = HoldingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
            heartbeat_interval=60,
        )
        await manager.start()
        started = await manager.start_benchmark()
        await executor.entered.wait()
        response = await manager.stream_events(started.run_id, None)
        active = manager._active
        assert active is not None
        iterator = response.body_iterator.__aiter__()
        waiting = asyncio.ensure_future(iterator.__anext__())
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert active.subscribers == set()
        executor.release.set()
        await wait_for_terminal(manager, started.run_id)

    asyncio.run(scenario())


def test_download_contains_only_public_run_artifacts(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        executor = CompletingExecutor()
        manager = BenchmarkManager(
            settings=settings,
            coordinator=WorkspaceOperationCoordinator(),
            executor=executor,
        )
        await manager.start()
        started = await manager.start_benchmark()
        executor.release.set()
        await wait_for_terminal(manager, started.run_id)
        run_dir = settings.benchmark_results_dir / str(started.run_id)
        (run_dir / "summary.json").write_text('{"public":true}', encoding="utf-8")
        (run_dir / "secret.txt").write_text("private", encoding="utf-8")
        nested = run_dir / "nested"
        nested.mkdir()
        (nested / "unrelated.json").write_text("{}", encoding="utf-8")

        response = await manager.download_benchmark(started.run_id)

        with zipfile.ZipFile(io.BytesIO(bytes(response.body))) as archive:
            assert sorted(archive.namelist()) == [
                "cases.jsonl",
                "events.jsonl",
                "run.json",
                "summary.json",
            ]
            assert archive.read("summary.json") == b'{"public":true}'
        assert response.media_type == "application/zip"
        assert response.headers["content-disposition"] == (
            f'attachment; filename="benchmark-{started.run_id}.zip"'
        )
        with pytest.raises(BenchmarkNotFoundError):
            await manager.download_benchmark(uuid4())

        outside = tmp_path / "outside-summary.json"
        outside.write_text('{"secret":true}', encoding="utf-8")
        (run_dir / "summary.json").unlink()
        try:
            (run_dir / "summary.json").symlink_to(outside)
        except OSError:
            return
        without_symlink = await manager.download_benchmark(started.run_id)
        with zipfile.ZipFile(io.BytesIO(bytes(without_symlink.body))) as archive:
            assert "summary.json" not in archive.namelist()

    asyncio.run(scenario())
