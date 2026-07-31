from pathlib import Path
from typing import Any

from fastapi import FastAPI

import modules.app as app_module
from modules.config import Settings


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        sources_dir=tmp_path / "sources",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        manifest_path=tmp_path / "data" / "manifest.json",
        trace_dir=tmp_path / "data" / "traces",
        logs_dir=tmp_path / "logs",
        benchmark_results_dir=tmp_path / "evals" / "results",
        server_host="0.0.0.0",
        server_port=8123,
        log_level="WARNING",
    )


def write_frontend_build(tmp_path: Path) -> Path:
    dist = tmp_path / "frontend" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<main>workspace</main>", encoding="utf-8")
    return dist


def test_create_application_composes_fastapi_without_constructing_services(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given a container factory that would expose eager construction
    calls = 0

    def container_factory():
        nonlocal calls
        calls += 1
        raise AssertionError("container must remain lazy")

    monkeypatch.setattr(app_module, "create_application_container", container_factory)

    # When the production FastAPI object is composed
    application = app_module.create_application(
        settings=make_settings(tmp_path),
        frontend_dist=write_frontend_build(tmp_path),
    )

    # Then service construction remains lifespan-owned
    assert isinstance(application, FastAPI)
    assert calls == 0


def test_main_launches_uvicorn_with_validated_server_settings(tmp_path: Path) -> None:
    # Given validated server settings and a captured server runner
    calls: list[tuple[FastAPI, dict[str, Any]]] = []

    def server_runner(application: FastAPI, **kwargs: Any) -> None:
        calls.append((application, kwargs))

    # When the production launcher runs
    result = app_module.main(
        settings=make_settings(tmp_path),
        frontend_dist=write_frontend_build(tmp_path),
        server_runner=server_runner,
    )

    # Then Uvicorn receives the validated network configuration
    assert result == 0
    assert len(calls) == 1
    assert isinstance(calls[0][0], FastAPI)
    assert calls[0][1] == {
        "host": "0.0.0.0",
        "port": 8123,
        "log_level": "warning",
    }


def test_main_reports_missing_frontend_build_without_starting_server(
    tmp_path: Path,
    capsys,
) -> None:
    # Given no production frontend build
    started = False

    def server_runner(_application: FastAPI, **_kwargs: Any) -> None:
        nonlocal started
        started = True

    # When the production launcher runs
    result = app_module.main(
        settings=make_settings(tmp_path),
        frontend_dist=tmp_path / "missing",
        server_runner=server_runner,
    )

    # Then startup stops with an actionable build instruction
    assert result == 1
    assert started is False
    assert "npm ci && npm run build" in capsys.readouterr().err
