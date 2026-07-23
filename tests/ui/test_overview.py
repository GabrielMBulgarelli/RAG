import json
from pathlib import Path
from typing import Any, cast

from modules.models import IngestionManifest, ManifestDocument, ReconciliationResult
from modules.ui.application import RAGApplication
from modules.ui.contracts import CorpusSnapshot


class OverviewManager:
    def __init__(self, root: Path):
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
        return ReconciliationResult(missing_source_files=[Path("manual.txt")])

    def chunk_count(self) -> int:
        return 2


def test_dashboard_snapshot_uses_existing_models_index_and_schema_v2_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Arrange
    result = tmp_path / "standard"
    result.mkdir()
    (result / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "result_kind": "standard_benchmark",
                "configuration": {
                    "dataset_name": "multihop",
                    "evaluated_split": "development",
                    "systems": ["dense", "bm25", "hybrid", "agentic"],
                    "timestamp": "2026-07-22T12:00:00+00:00",
                },
                "metrics": {
                    "dense": {
                        "recall_at_5": {
                            "value": 0.75,
                            "status": "measured",
                            "sample_count": 8,
                            "note": "Existing observation.",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (result / "cases.jsonl").write_text("{}\n" * 8, encoding="utf-8")
    application = RAGApplication(vector_db=cast(Any, OverviewManager(tmp_path)))
    monkeypatch.setattr(
        application,
        "_ollama_info",
        lambda: {
            "reachable": True,
            "models": ["qwen3.5:9b", "nomic-embed-text:latest"],
        },
    )
    monkeypatch.setattr(application, "latest_evaluation", lambda: result)

    # Act
    snapshot = application.dashboard_snapshot()
    latest = application.latest_evaluation_summary()

    # Assert the snapshot preserves backend facts without inventing aggregates.
    assert snapshot.runtime.chat_model == "qwen3.5:9b"
    assert snapshot.runtime.embedding_model == "nomic-embed-text"
    assert snapshot.corpus == CorpusSnapshot(1, 3, 2, "ready")
    assert snapshot.index.missing_source_file_count == 1
    assert snapshot.evaluation is not None
    assert latest == snapshot.evaluation
    assert snapshot.evaluation.result_kind == "standard"
    retrieval = snapshot.evaluation.quality_categories[0]
    assert retrieval.name == "Retrieval"
    assert retrieval.metrics[0].observations[0].value == 0.75
    assert {category.name for category in snapshot.evaluation.quality_categories} == {
        "Retrieval",
        "Evidence and grounding",
        "Answer quality",
        "Workflow cost",
    }


def test_blocked_runtime_still_reports_configured_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Arrange
    application = RAGApplication(vector_db=cast(Any, OverviewManager(tmp_path)))
    monkeypatch.setattr(
        application,
        "_ollama_info",
        lambda: {"reachable": False, "models": []},
    )
    monkeypatch.setattr(application, "latest_evaluation", lambda: None)

    # Act
    runtime = application.runtime_snapshot()

    # Assert blocked readiness still identifies the configured recovery targets.
    assert runtime.state == "blocked"
    assert runtime.chat_model == "qwen3.5:9b"
    assert runtime.embedding_model == "nomic-embed-text"
    assert "ollama serve" in runtime.detail
    assert "ollama pull qwen3.5:9b" in runtime.detail
    assert "ollama pull nomic-embed-text" in runtime.detail
