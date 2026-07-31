from pathlib import Path

from fastapi.testclient import TestClient

from modules.api.app import create_app
from modules.application.benchmark_manager import BenchmarkManager
from modules.application.workspace_service import WorkspaceService
from modules.bootstrap import create_application_container
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
    )


def test_production_container_composes_workspace_with_real_benchmarks(tmp_path: Path) -> None:
    # When the production container is composed
    container = create_application_container(make_settings(tmp_path))

    # Then both services share the production operation coordinator
    assert isinstance(container.workspace, WorkspaceService)
    assert isinstance(container.benchmarks, BenchmarkManager)
    assert container.workspace._benchmark_available is True
    assert container.workspace.coordinator is container.benchmarks.coordinator


def test_production_benchmark_endpoint_starts_a_durable_run(tmp_path: Path) -> None:
    # Given the production dependency composition
    settings = make_settings(tmp_path)
    app = create_app(lambda: create_application_container(settings))

    # When the benchmark endpoint is invoked
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/benchmarks")

    # Then the API accepts the run through the real benchmark manager
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["links"]["run"].startswith("/api/benchmarks/")
