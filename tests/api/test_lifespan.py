import importlib
import sys
from typing import cast

import pytest
from fastapi.testclient import TestClient

from modules.api.app import create_app
from modules.api.dependencies import (
    ApplicationContainer,
    BenchmarkManager,
    WorkspaceService,
)


class LifecycleFake:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.start_count = 0
        self.close_count = 0

    async def start(self) -> None:
        self.start_count += 1
        self.events.append(f"{self.name}.start")

    async def close(self) -> None:
        self.close_count += 1
        self.events.append(f"{self.name}.close")


class FailingStartFake(LifecycleFake):
    async def start(self) -> None:
        await super().start()
        raise RuntimeError("startup failed")


def test_lifespan_constructs_starts_and_closes_owned_dependencies_once() -> None:
    events: list[str] = []
    workspace = LifecycleFake("workspace", events)
    benchmarks = LifecycleFake("benchmarks", events)
    factory_calls = 0

    def container_factory() -> ApplicationContainer:
        nonlocal factory_calls
        factory_calls += 1
        return ApplicationContainer(
            workspace=cast(WorkspaceService, workspace),
            benchmarks=cast(BenchmarkManager, benchmarks),
        )

    app = create_app(container_factory)

    assert factory_calls == 0
    with TestClient(app):
        assert factory_calls == 1
        assert workspace.start_count == 1
        assert benchmarks.start_count == 1
        assert app.state.container.workspace is workspace

    assert workspace.close_count == 1
    assert benchmarks.close_count == 1
    assert events == [
        "workspace.start",
        "benchmarks.start",
        "benchmarks.close",
        "workspace.close",
    ]


def test_importing_app_module_does_not_construct_container() -> None:
    module_name = "modules.api.app"
    imported = sys.modules.pop(module_name, None)
    try:
        importlib.import_module(module_name)
    finally:
        if imported is not None:
            sys.modules[module_name] = imported


def test_startup_failure_does_not_publish_container_and_closes_started_dependency() -> None:
    events: list[str] = []
    workspace = LifecycleFake("workspace", events)
    benchmarks = FailingStartFake("benchmarks", events)
    app = create_app(
        lambda: ApplicationContainer(
            workspace=cast(WorkspaceService, workspace),
            benchmarks=cast(BenchmarkManager, benchmarks),
        )
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(app):
            pass

    assert not hasattr(app.state, "container")
    assert workspace.close_count == 1
    assert events == ["workspace.start", "benchmarks.start", "workspace.close"]
