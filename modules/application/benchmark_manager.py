"""Persistent neutral benchmark run ownership."""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import zipfile
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import JsonValue
from starlette.responses import Response, StreamingResponse

from modules.application.errors import BenchmarkNotFoundError
from modules.application.models import (
    ApiProblem,
    BenchmarkCaseDetail,
    BenchmarkEvent,
    BenchmarkEventType,
    BenchmarkFailure,
    BenchmarkMetadata,
    BenchmarkProgress,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkSection,
    BenchmarkStartResponse,
    OperationKind,
    ResourceLinks,
)
from modules.application.operation_coordinator import (
    OperationLease,
    WorkspaceOperationCoordinator,
)
from modules.config import Settings, config

_TERMINAL_STATUSES = {
    BenchmarkRunStatus.CANCELLED,
    BenchmarkRunStatus.COMPLETED,
    BenchmarkRunStatus.FAILED,
}
_EXECUTOR_EVENT_TYPES = {
    BenchmarkEventType.SYSTEM_STARTED,
    BenchmarkEventType.CASE_STARTED,
    BenchmarkEventType.CASE_COMPLETED,
    BenchmarkEventType.CASE_FAILED,
    BenchmarkEventType.SYSTEM_COMPLETED,
}


class BenchmarkCancellation:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def request(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


@dataclass(frozen=True)
class BenchmarkExecutionResult:
    sections: Sequence[BenchmarkSection]
    failures: Sequence[BenchmarkFailure]


class BenchmarkReporter(Protocol):
    async def publish(
        self,
        event_type: BenchmarkEventType,
        data: dict[str, JsonValue],
        *,
        progress: BenchmarkProgress,
        case: BenchmarkCaseDetail | None = None,
    ) -> None: ...


class BenchmarkExecutor(Protocol):
    def initial_metadata(self) -> BenchmarkMetadata: ...

    async def execute(
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult: ...


@dataclass
class _ActiveRun:
    run: BenchmarkRun
    lease: OperationLease
    cancellation: BenchmarkCancellation
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    events: deque[BenchmarkEvent] = field(default_factory=lambda: deque(maxlen=512))
    subscribers: set[asyncio.Queue[BenchmarkEvent | None]] = field(default_factory=set)
    task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    heartbeat_stop: asyncio.Event = field(default_factory=asyncio.Event)


class _RunReporter:
    def __init__(self, manager: BenchmarkManager, active: _ActiveRun) -> None:
        self._manager = manager
        self._active = active

    async def publish(
        self,
        event_type: BenchmarkEventType,
        data: dict[str, JsonValue],
        *,
        progress: BenchmarkProgress,
        case: BenchmarkCaseDetail | None = None,
    ) -> None:
        if event_type not in _EXECUTOR_EVENT_TYPES:
            raise ValueError("benchmark lifecycle events are manager-owned")
        await self._manager._publish(
            self._active,
            event_type,
            data,
            progress=progress,
            case=case,
        )


class BenchmarkManager:
    def __init__(
        self,
        *,
        executor: BenchmarkExecutor,
        settings: Settings | None = None,
        coordinator: WorkspaceOperationCoordinator | None = None,
        heartbeat_interval: float = 15.0,
    ) -> None:
        self.settings = settings or config
        self.coordinator = coordinator or WorkspaceOperationCoordinator()
        self._executor = executor
        self._heartbeat_interval = heartbeat_interval
        self._started = False
        self._closed = False
        self._active: _ActiveRun | None = None

    async def start(self) -> None:
        if self._started:
            return
        await asyncio.to_thread(
            self.settings.benchmark_results_dir.mkdir, parents=True, exist_ok=True
        )
        await asyncio.to_thread(self._recover_runs)
        self._started = True
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        active = self._active
        if active is not None and active.task is not None:
            await self.cancel_benchmark(active.run.run_id)
            await active.task

    @staticmethod
    def _links(run_id: UUID) -> ResourceLinks:
        return ResourceLinks(
            run=f"/api/benchmarks/{run_id}",
            events=f"/api/benchmarks/{run_id}/events",
            download=f"/api/benchmarks/{run_id}/download",
        )

    def _run_dir(self, run_id: UUID, *, must_exist: bool = False) -> Path:
        root = self.settings.benchmark_results_dir.resolve()
        candidate = self.settings.benchmark_results_dir / str(run_id)
        if candidate.is_symlink():
            raise BenchmarkNotFoundError()
        resolved = candidate.resolve()
        if resolved.parent != root:
            raise BenchmarkNotFoundError()
        if must_exist and not resolved.is_dir():
            raise BenchmarkNotFoundError()
        return resolved

    def _artifact_path(
        self,
        run_id: UUID,
        name: str,
        *,
        must_exist: bool = False,
    ) -> Path:
        run_dir = self._run_dir(run_id, must_exist=must_exist)
        candidate = run_dir / name
        if candidate.is_symlink():
            raise BenchmarkNotFoundError()
        resolved = candidate.resolve()
        if resolved.parent != run_dir:
            raise BenchmarkNotFoundError()
        if must_exist and not resolved.is_file():
            raise BenchmarkNotFoundError()
        return resolved

    def _write_run(self, run: BenchmarkRun) -> None:
        run_dir = self._run_dir(run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        target = self._artifact_path(run.run_id, "run.json")
        temporary = self._artifact_path(run.run_id, "run.tmp")
        temporary.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        BenchmarkRun.model_validate_json(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, target)

    def _append_event(self, event: BenchmarkEvent) -> None:
        path = self._artifact_path(event.run_id, "events.jsonl")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")
            stream.flush()

    def _append_case(self, run_id: UUID, case: BenchmarkCaseDetail) -> None:
        path = self._artifact_path(run_id, "cases.jsonl")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(case.model_dump_json() + "\n")
            stream.flush()

    def _read_run_snapshot(self, run_id: UUID) -> BenchmarkRun:
        try:
            run = BenchmarkRun.model_validate_json(
                self._artifact_path(run_id, "run.json", must_exist=True).read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise BenchmarkNotFoundError() from exc
        if run.run_id != run_id:
            raise BenchmarkNotFoundError()
        return run

    def _read_run(self, run_id: UUID) -> BenchmarkRun:
        run = self._read_run_snapshot(run_id)
        self._read_events(run_id)
        self._read_cases(run_id)
        return run

    def _read_events(self, run_id: UUID) -> list[BenchmarkEvent]:
        try:
            events = [
                BenchmarkEvent.model_validate_json(line)
                for line in self._artifact_path(
                    run_id,
                    "events.jsonl",
                    must_exist=True,
                )
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        except Exception as exc:
            raise BenchmarkNotFoundError() from exc
        if any(
            event.run_id != run_id or event.event_id != index
            for index, event in enumerate(events, start=1)
        ):
            raise BenchmarkNotFoundError()
        return events

    def _read_cases(self, run_id: UUID) -> list[BenchmarkCaseDetail]:
        try:
            return [
                BenchmarkCaseDetail.model_validate_json(line)
                for line in self._artifact_path(
                    run_id,
                    "cases.jsonl",
                    must_exist=True,
                )
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        except Exception as exc:
            raise BenchmarkNotFoundError() from exc

    def _recover_runs(self) -> None:
        for run_dir in self.settings.benchmark_results_dir.iterdir():
            if run_dir.is_symlink() or not run_dir.is_dir():
                continue
            try:
                run_id = UUID(run_dir.name)
                run = self._read_run_snapshot(run_id)
            except (ValueError, BenchmarkNotFoundError):
                continue
            if run.status not in {
                BenchmarkRunStatus.QUEUED,
                BenchmarkRunStatus.RUNNING,
                BenchmarkRunStatus.CANCELLATION_REQUESTED,
            }:
                continue
            try:
                events = self._read_events(run_id)
            except BenchmarkNotFoundError:
                events = []
                try:
                    self._artifact_path(run_id, "events.jsonl").write_text(
                        "",
                        encoding="utf-8",
                    )
                except BenchmarkNotFoundError:
                    continue
            run = run.model_copy(
                deep=True,
                update={
                    "status": BenchmarkRunStatus.FAILED,
                    "metadata": run.metadata.model_copy(
                        update={"completed_at": datetime.now(timezone.utc)}
                    ),
                    "error": ApiProblem(
                        code="benchmark_interrupted",
                        message="The benchmark was interrupted by a previous process.",
                        details={},
                    ),
                },
            )
            self._write_run(run)
            event = BenchmarkEvent(
                event_id=events[-1].event_id + 1 if events else 1,
                run_id=run_id,
                type=BenchmarkEventType.BENCHMARK_FAILED,
                timestamp=datetime.now(timezone.utc),
                data={},
            )
            self._append_event(event)

    async def start_benchmark(self) -> BenchmarkStartResponse:
        if not self._started:
            await self.start()
        run_id = uuid4()
        lease = self.coordinator.acquire(
            OperationKind.BENCHMARK,
            benchmark_run_id=run_id,
        )
        run_dir = self._run_dir(run_id)
        try:
            metadata = self._executor.initial_metadata().model_copy(
                deep=True,
                update={"started_at": None, "completed_at": None},
            )
            run = BenchmarkRun(
                run_id=run_id,
                status=BenchmarkRunStatus.QUEUED,
                progress=BenchmarkProgress(
                    completed_cases=0,
                    total_cases=0,
                    total_systems=len(metadata.systems),
                ),
                metadata=metadata,
                sections=[],
                failures=[],
                links=self._links(run_id),
                error=None,
            )
            active = _ActiveRun(
                run=run,
                lease=lease,
                cancellation=BenchmarkCancellation(),
            )
            await asyncio.to_thread(run_dir.mkdir, parents=True, exist_ok=False)
            await asyncio.to_thread(self._write_run, run)
            await asyncio.to_thread((run_dir / "events.jsonl").touch)
            await asyncio.to_thread((run_dir / "cases.jsonl").touch)
            self._active = active
            active.task = asyncio.create_task(
                self._execute(active),
                name=f"benchmark-{run_id}",
            )
        except Exception:
            self._active = None
            lease.release()
            try:
                await asyncio.to_thread(shutil.rmtree, run_dir)
            except Exception:
                pass
            raise
        return BenchmarkStartResponse(
            run_id=run_id,
            status=BenchmarkRunStatus.QUEUED,
            links=run.links.model_copy(deep=True),
        )

    async def _execute(self, active: _ActiveRun) -> None:
        try:
            async with active.lock:
                cancelled_while_queued = active.cancellation.is_cancelled
                if not cancelled_while_queued:
                    await self._transition_locked(
                        active,
                        update={
                            "status": BenchmarkRunStatus.RUNNING,
                            "metadata": active.run.metadata.model_copy(
                                update={"started_at": datetime.now(timezone.utc)}
                            ),
                        },
                        event_type=BenchmarkEventType.BENCHMARK_STARTED,
                    )
            if cancelled_while_queued:
                await self._finish_cancelled(active)
                return
            active.heartbeat_task = asyncio.create_task(self._heartbeat(active))
            result = await self._executor.execute(
                active.run.run_id,
                _RunReporter(self, active),
                active.cancellation,
            )
            await self._stop_heartbeat(active)
            async with active.lock:
                if active.cancellation.is_cancelled:
                    await self._finish_cancelled_locked(active)
                else:
                    completed_at = datetime.now(timezone.utc)
                    await self._transition_locked(
                        active,
                        update={
                            "status": BenchmarkRunStatus.COMPLETED,
                            "sections": list(result.sections),
                            "failures": list(result.failures),
                            "metadata": active.run.metadata.model_copy(
                                update={"completed_at": completed_at}
                            ),
                        },
                        event_type=BenchmarkEventType.BENCHMARK_COMPLETED,
                    )
        except Exception:
            await self._stop_heartbeat(active)
            async with active.lock:
                if active.cancellation.is_cancelled:
                    await self._finish_cancelled_locked(active)
                else:
                    await self._transition_locked(
                        active,
                        update={
                            "status": BenchmarkRunStatus.FAILED,
                            "metadata": active.run.metadata.model_copy(
                                update={"completed_at": datetime.now(timezone.utc)}
                            ),
                            "error": ApiProblem(
                                code="benchmark_failed",
                                message="The benchmark could not be completed.",
                                details={},
                            ),
                        },
                        event_type=BenchmarkEventType.BENCHMARK_FAILED,
                    )
        finally:
            await self._close_subscribers(active)
            active.lease.release()
            if self._active is active:
                self._active = None

    async def _heartbeat(self, active: _ActiveRun) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    active.heartbeat_stop.wait(),
                    timeout=self._heartbeat_interval,
                )
                return
            except TimeoutError:
                if active.heartbeat_stop.is_set():
                    return
            await self._publish(
                active,
                BenchmarkEventType.HEARTBEAT,
                {},
                progress=active.run.progress,
            )

    @staticmethod
    async def _stop_heartbeat(active: _ActiveRun) -> None:
        task = active.heartbeat_task
        if task is None:
            return
        active.heartbeat_stop.set()
        await task
        active.heartbeat_task = None

    async def _finish_cancelled(self, active: _ActiveRun) -> None:
        async with active.lock:
            await self._finish_cancelled_locked(active)

    async def _finish_cancelled_locked(self, active: _ActiveRun) -> None:
        await self._transition_locked(
            active,
            update={
                "status": BenchmarkRunStatus.CANCELLED,
                "metadata": active.run.metadata.model_copy(
                    update={"completed_at": datetime.now(timezone.utc)}
                ),
            },
            event_type=BenchmarkEventType.BENCHMARK_CANCELLED,
        )

    async def _transition_locked(
        self,
        active: _ActiveRun,
        *,
        update: dict[str, object],
        event_type: BenchmarkEventType,
    ) -> BenchmarkEvent:
        active.run = active.run.model_copy(deep=True, update=update)
        event = await self._publish_locked(
            active,
            event_type,
            {},
            progress=active.run.progress,
            case=None,
        )
        if active.run.status in _TERMINAL_STATUSES:
            active.lease.release()
        return event

    async def _publish(
        self,
        active: _ActiveRun,
        event_type: BenchmarkEventType,
        data: dict[str, JsonValue],
        *,
        progress: BenchmarkProgress,
        case: BenchmarkCaseDetail | None = None,
    ) -> BenchmarkEvent:
        async with active.lock:
            return await self._publish_locked(
                active,
                event_type,
                data,
                progress=progress,
                case=case,
            )

    async def _publish_locked(
        self,
        active: _ActiveRun,
        event_type: BenchmarkEventType,
        data: dict[str, JsonValue],
        *,
        progress: BenchmarkProgress,
        case: BenchmarkCaseDetail | None,
    ) -> BenchmarkEvent:
        active.run = active.run.model_copy(
            deep=True,
            update={"progress": progress.model_copy(deep=True)},
        )
        if case is not None:
            await asyncio.to_thread(self._append_case, active.run.run_id, case)
        await asyncio.to_thread(self._write_run, active.run)
        event = BenchmarkEvent(
            event_id=active.events[-1].event_id + 1 if active.events else 1,
            run_id=active.run.run_id,
            type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data,
        )
        await asyncio.to_thread(self._append_event, event)
        active.events.append(event)
        self._broadcast(active, event)
        return event.model_copy(deep=True)

    async def get_benchmark(self, run_id: UUID) -> BenchmarkRun:
        active = self._active
        if active is not None and active.run.run_id == run_id:
            async with active.lock:
                return active.run.model_copy(deep=True)
        return (await asyncio.to_thread(self._read_run, run_id)).model_copy(deep=True)

    async def latest_benchmark(self) -> BenchmarkRun:
        def load() -> BenchmarkRun:
            completed: list[BenchmarkRun] = []
            if not self.settings.benchmark_results_dir.is_dir():
                raise BenchmarkNotFoundError()
            for run_dir in self.settings.benchmark_results_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                try:
                    run = self._read_run(UUID(run_dir.name))
                except (ValueError, BenchmarkNotFoundError):
                    continue
                if (
                    run.status is BenchmarkRunStatus.COMPLETED
                    and run.metadata.completed_at is not None
                ):
                    completed.append(run)
            if not completed:
                raise BenchmarkNotFoundError()
            return max(
                completed,
                key=lambda item: (
                    item.metadata.completed_at or datetime.min.replace(tzinfo=timezone.utc)
                ),
            )

        return (await asyncio.to_thread(load)).model_copy(deep=True)

    async def cancel_benchmark(self, run_id: UUID) -> BenchmarkRun:
        active = self._active
        if active is None or active.run.run_id != run_id:
            run = await self.get_benchmark(run_id)
            return run
        async with active.lock:
            if active.run.status in _TERMINAL_STATUSES:
                return active.run.model_copy(deep=True)
            if active.run.status is BenchmarkRunStatus.CANCELLATION_REQUESTED:
                return active.run.model_copy(deep=True)
            active.cancellation.request()
            self.coordinator.request_benchmark_cancellation(run_id)
            active.run = active.run.model_copy(
                deep=True,
                update={"status": BenchmarkRunStatus.CANCELLATION_REQUESTED},
            )
            await asyncio.to_thread(self._write_run, active.run)
            event = BenchmarkEvent(
                event_id=active.events[-1].event_id + 1 if active.events else 1,
                run_id=run_id,
                type=BenchmarkEventType.BENCHMARK_CANCELLATION_REQUESTED,
                timestamp=datetime.now(timezone.utc),
                data={},
            )
            await asyncio.to_thread(self._append_event, event)
            active.events.append(event)
            self._broadcast(active, event)
            return active.run.model_copy(deep=True)

    async def get_case(
        self,
        run_id: UUID,
        case_id: str,
        system_id: str,
    ) -> BenchmarkCaseDetail:
        await self.get_benchmark(run_id)

        def load() -> BenchmarkCaseDetail:
            found: BenchmarkCaseDetail | None = None
            for case in self._read_cases(run_id):
                if case.case_id == case_id and case.system == system_id:
                    found = case
            if found is None:
                raise BenchmarkNotFoundError()
            return found

        return (await asyncio.to_thread(load)).model_copy(deep=True)

    async def download_benchmark(self, run_id: UUID) -> Response:
        await self.get_benchmark(run_id)

        def build() -> bytes:
            output = io.BytesIO()
            with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in (
                    "run.json",
                    "events.jsonl",
                    "cases.jsonl",
                    "summary.json",
                ):
                    try:
                        artifact = self._artifact_path(run_id, name, must_exist=True)
                    except BenchmarkNotFoundError:
                        continue
                    archive.writestr(name, artifact.read_bytes())
            return output.getvalue()

        payload = await asyncio.to_thread(build)
        return Response(
            payload,
            media_type="application/zip",
            headers={"Content-Disposition": (f'attachment; filename="benchmark-{run_id}.zip"')},
        )

    @staticmethod
    def _broadcast(active: _ActiveRun, event: BenchmarkEvent) -> None:
        for subscriber in tuple(active.subscribers):
            try:
                subscriber.put_nowait(event.model_copy(deep=True))
            except asyncio.QueueFull:
                active.subscribers.discard(subscriber)
                while not subscriber.empty():
                    try:
                        subscriber.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                subscriber.put_nowait(None)

    @staticmethod
    async def _close_subscribers(active: _ActiveRun) -> None:
        async with active.lock:
            for subscriber in tuple(active.subscribers):
                active.subscribers.discard(subscriber)
                if subscriber.full():
                    try:
                        subscriber.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                subscriber.put_nowait(None)

    @staticmethod
    def _sse(event: BenchmarkEvent) -> bytes:
        return (
            f"id: {event.event_id}\nevent: {event.type.value}\ndata: {event.model_dump_json()}\n\n"
        ).encode()

    async def stream_events(
        self,
        run_id: UUID,
        last_event_id: int | None,
    ) -> StreamingResponse:
        run = await self.get_benchmark(run_id)
        active = self._active
        if active is None or active.run.run_id != run_id or run.status in _TERMINAL_STATUSES:
            replay = (
                []
                if last_event_id is None
                else [
                    event
                    for event in await asyncio.to_thread(self._read_events, run_id)
                    if event.event_id > last_event_id
                ]
            )

            async def terminal_stream():
                for event in replay:
                    yield self._sse(event)

            return StreamingResponse(terminal_stream(), media_type="text/event-stream")

        async with active.lock:
            boundary = active.events[-1].event_id if active.events else 0
            if last_event_id is None:
                replay = []
            elif active.events and last_event_id >= active.events[0].event_id - 1:
                replay = [
                    event.model_copy(deep=True)
                    for event in active.events
                    if event.event_id > last_event_id
                ]
            else:
                replay = [
                    event
                    for event in await asyncio.to_thread(self._read_events, run_id)
                    if last_event_id < event.event_id <= boundary
                ]
            if active.run.status in _TERMINAL_STATUSES:
                subscriber = None
            else:
                subscriber = asyncio.Queue(maxsize=64)
                active.subscribers.add(subscriber)

        if subscriber is None:

            async def raced_terminal_stream():
                for event in replay:
                    yield self._sse(event)

            return StreamingResponse(
                raced_terminal_stream(),
                media_type="text/event-stream",
            )

        async def live_stream():
            live_after = max(boundary, last_event_id or 0)
            try:
                for event in replay:
                    yield self._sse(event)
                while True:
                    event = await subscriber.get()
                    if event is None:
                        return
                    if event.event_id > live_after:
                        yield self._sse(event)
            finally:
                async with active.lock:
                    active.subscribers.discard(subscriber)

        return StreamingResponse(live_stream(), media_type="text/event-stream")
