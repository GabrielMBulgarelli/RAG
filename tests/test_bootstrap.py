import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from modules.api.app import create_app
from modules.application import full_rag_benchmark
from modules.application.benchmark_manager import BenchmarkManager
from modules.application.workspace_service import WorkspaceService
from modules.bootstrap import create_application_container
from modules.config import Settings
from modules.evaluation_models import CANONICAL_BENCHMARK_CASE_IDS, EvaluationCase


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


def write_canonical_benchmark_cases(benchmark_root: Path) -> None:
    cases = [
        EvaluationCase(
            id=case_id,
            split="development",
            category="comparison",
            question="Which organization changed its policy?",
            answerable=True,
            relevant_chunk_ids=["chunk-1"],
            relevant_document_ids=["document-1"],
            expected_answer="Example Organization",
            expected_route="complex_search",
            expected_strategy="hybrid",
            expected_retry=False,
            expected_conflict=False,
        )
        for case_id in CANONICAL_BENCHMARK_CASE_IDS
    ]
    benchmark_root.mkdir(parents=True)
    (benchmark_root / "cases.jsonl").write_text(
        "".join(case.model_dump_json() + "\n" for case in cases),
        encoding="utf-8",
    )


def test_production_container_composes_workspace_with_real_benchmarks(tmp_path: Path) -> None:
    # When the production container is composed
    container = create_application_container(make_settings(tmp_path))

    # Then both services share the production operation coordinator
    assert isinstance(container.workspace, WorkspaceService)
    assert isinstance(container.benchmarks, BenchmarkManager)
    assert container.workspace.coordinator is container.benchmarks.coordinator
    assert (
        container.workspace._completed_benchmark_probe
        == container.benchmarks.has_completed_benchmark
    )


def test_production_composition_has_no_placeholder_benchmark_module() -> None:
    obsolete_module = "modules.application." + "unavailable_" + "benchmarks"

    assert importlib.util.find_spec(obsolete_module) is None


def test_production_benchmark_endpoint_starts_a_durable_run(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    # Given the production dependency composition
    benchmark_root = tmp_path / "multihop"
    write_canonical_benchmark_cases(benchmark_root)
    monkeypatch.setattr(full_rag_benchmark, "MULTIHOP_ROOT", benchmark_root)
    settings = make_settings(tmp_path)
    app = create_app(lambda: create_application_container(settings))

    # When the benchmark endpoint is invoked
    with TestClient(app) as client:
        response = client.post("/api/benchmarks")

    # Then the API accepts the run through the real benchmark manager
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["links"]["run"].startswith("/api/benchmarks/")
