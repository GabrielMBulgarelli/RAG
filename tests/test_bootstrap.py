from pathlib import Path

from fastapi.testclient import TestClient

from modules.api.app import create_app
from modules.application.unavailable_benchmarks import UnavailableBenchmarkManager
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


def test_production_container_uses_workspace_with_unavailable_benchmarks(tmp_path: Path) -> None:
    # When the production container is composed
    container = create_application_container(make_settings(tmp_path))

    # Then it owns the workspace and explicit unavailable adapter
    assert isinstance(container.workspace, WorkspaceService)
    assert isinstance(container.benchmarks, UnavailableBenchmarkManager)
    assert container.workspace._benchmark_available is False


def test_production_benchmark_endpoint_reports_unavailable(tmp_path: Path) -> None:
    # Given the production dependency composition
    settings = make_settings(tmp_path)
    app = create_app(lambda: create_application_container(settings))

    # When the benchmark endpoint is invoked
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/benchmarks")

    # Then the API reports the temporary capability boundary
    assert response.status_code == 503
    assert response.json() == {
        "code": "benchmark_unavailable",
        "message": "Benchmark execution is not configured.",
        "details": {},
    }
