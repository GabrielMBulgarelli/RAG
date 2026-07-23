from pathlib import Path
from typing import Any, cast

import gradio as gr

from modules.models import IngestionManifest, ManifestDocument, ReconciliationResult
from modules.ui.application import EvaluationReadiness, RAGApplication
from modules.ui.contracts import (
    SafeConfigurationValue,
    SystemCheck,
    SystemPageSnapshot,
)
from modules.ui.pages.system import build_system_page, system_values
from modules.ui.presenters import render_system_page


class SystemManager:
    def __init__(self, root: Path) -> None:
        self.settings = cast(Any, type("Settings", (), {"sources_dir": root})())
        self.record = ManifestDocument(
            document_id="private-id",
            relative_path="manual.txt",
            filename="manual.txt",
            content_hash="abcdef",
            chunk_ids=["chunk-1", "chunk-2"],
            page_count=3,
            chunk_count=2,
            embedding_model="nomic-embed-text",
            chunk_size=700,
            chunk_overlap=100,
        )

    def manifest(self) -> IngestionManifest:
        return IngestionManifest(documents={"private-id": self.record})

    def reconcile_index(self) -> ReconciliationResult:
        return ReconciliationResult(missing_source_files=[Path("private/manual.txt")])

    def chunk_count(self) -> int:
        return 2


def _system_snapshot(*, blocked: bool = False) -> SystemPageSnapshot:
    runtime_state = "blocked" if blocked else "ready"
    runtime_detail = (
        "Start Ollama at http://localhost:11434."
        if blocked
        else "Ollama, required models, index, and application services are available."
    )
    return SystemPageSnapshot(
        state=runtime_state,
        title="Unavailable" if blocked else "Ready",
        detail=runtime_detail,
        can_load_models=not blocked,
        runtime_checks=(
            SystemCheck(
                "runtime",
                "Ollama connectivity",
                runtime_state,
                "<offline>" if blocked else "Reachable",
            ),
            SystemCheck("models", "Chat model", "ready", "qwen3.5:9b"),
            SystemCheck("models", "Embedding model", "ready", "nomic-embed-text"),
            SystemCheck("runtime", "AI service initialization", "not_loaded", "Not loaded"),
        ),
        index_checks=(
            SystemCheck("index", "Chroma collection", "ready", "2 chunks"),
            SystemCheck("index", "Manifest", "ready", "1 document"),
            SystemCheck("index", "Missing chunks", "ready", "0"),
            SystemCheck("index", "Orphan chunks", "ready", "0"),
            SystemCheck("index", "Duplicate IDs", "ready", "0"),
            SystemCheck("index", "Missing source files", "review", "1"),
            SystemCheck("index", "Index compatibility", "ready", "Compatible"),
        ),
        evaluation_checks=(
            SystemCheck("evaluation", "Dataset readiness", "ready", "Ready"),
            SystemCheck(
                "evaluation",
                "Required models",
                "ready",
                "nomic-embed-text:latest, qwen3.5:9b",
            ),
            SystemCheck(
                "evaluation",
                "Latest compatible standard result",
                "not_loaded",
                "No stored result",
            ),
        ),
        safe_configuration=(
            SafeConfigurationValue("Ollama base URL", "http://localhost:11434"),
            SafeConfigurationValue("Chat model", "qwen3.5:9b"),
            SafeConfigurationValue("Embedding model", "nomic-embed-text"),
            SafeConfigurationValue("Gradio", "127.0.0.1:7860"),
            SafeConfigurationValue("Chunk size / overlap", "700 / 100"),
            SafeConfigurationValue("Retrieval candidates", "semantic 10 · sparse 10 · max 20"),
            SafeConfigurationValue("Context / subqueries / retries", "6 / 4 / 1"),
        ),
    )


def test_system_snapshot_groups_diagnostics_without_private_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Arrange local diagnostic dependencies.
    application = RAGApplication(vector_db=cast(Any, SystemManager(tmp_path)))
    monkeypatch.setattr(
        application,
        "_ollama_info",
        lambda: {
            "reachable": True,
            "models": ["qwen3.5:9b", "nomic-embed-text:latest"],
        },
    )
    monkeypatch.setattr(
        application,
        "evaluation_readiness",
        lambda _split, _systems, **_kwargs: EvaluationReadiness(
            state="ready",
            systems=("dense", "bm25", "hybrid", "agentic"),
            split="development",
        ),
    )
    monkeypatch.setattr(application, "latest_evaluation", lambda: None)

    # Act through the public snapshot boundary.
    snapshot = application.system_snapshot()

    # Assert only grouped, sanitized values cross that boundary.
    assert snapshot.state == "review"
    assert {check.name for check in snapshot.runtime_checks} == {
        "Ollama connectivity",
        "Chat model",
        "Embedding model",
        "AI service initialization",
    }
    assert {check.name for check in snapshot.index_checks} == {
        "Chroma collection",
        "Manifest",
        "Missing chunks",
        "Orphan chunks",
        "Duplicate IDs",
        "Missing source files",
        "Index compatibility",
    }
    assert {check.name for check in snapshot.evaluation_checks} == {
        "Dataset readiness",
        "Required models",
        "Latest compatible standard result",
    }
    rendered_values = " ".join(item.value for item in snapshot.safe_configuration)
    assert str(tmp_path) not in rendered_values
    assert "private/manual.txt" not in rendered_values
    assert "prompt" not in snapshot.__dataclass_fields__
    assert "reasoning" not in snapshot.__dataclass_fields__


def test_system_presenter_uses_blocking_banner_and_escapes_check_details() -> None:
    # Act with an unsafe diagnostic detail.
    rendered = render_system_page(_system_snapshot(blocked=True))

    # Assert blocking status is prominent and dynamic content is escaped.
    assert "Unavailable" in rendered
    assert "Start Ollama at http://localhost:11434." in rendered
    assert all(
        heading in rendered
        for heading in ("AI runtime", "Document index", "Evaluation", "Safe configuration")
    )
    assert "&lt;offline&gt;" in rendered
    assert "<offline>" not in rendered
    assert "/home/" not in rendered
    assert "RAG_" not in rendered


def test_system_page_exposes_actions_sections_and_fresh_snapshot_updates() -> None:
    # Arrange a controller-free page stub.
    class StubApplication:
        def system_snapshot(self) -> SystemPageSnapshot:
            return _system_snapshot()

        def load_ai_models(self) -> None:
            return None

    # Act by constructing the routed page and its initial refresh values.
    with gr.Blocks() as interface:
        build_system_page(cast(Any, StubApplication()))

    config = interface.get_config_file()
    values = {
        component["props"].get("value")
        for component in config["components"]
        if isinstance(component["props"].get("value"), str)
    }
    element_ids = {
        component["props"].get("elem_id")
        for component in config["components"]
        if component["props"].get("elem_id")
    }

    # Assert both manual controls and refresh outputs remain discoverable.
    assert "# System" in values
    assert {"Load AI Models", "Refresh"} <= values
    assert {
        "system-status",
        "system-diagnostics",
        "system-actions",
    } <= element_ids

    content, load_models = system_values(_system_snapshot())
    assert "AI runtime" in content
    assert load_models.get("visible") is True
    assert load_models.get("interactive") is True
