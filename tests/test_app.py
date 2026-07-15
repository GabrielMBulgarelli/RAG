import json
import os
from pathlib import Path

import modules.app as app_module
from modules.app import RAGApplication
from modules.models import (
    IngestionManifest,
    IngestionResult,
    ManifestDocument,
    ReconciliationResult,
)


class FakeManager:
    def __init__(self, root: Path):
        self.settings = type("Settings", (), {"sources_dir": root})()
        self.deleted: list[str] = []
        self.indexed: list[Path] = []
        self.rebuilt = False
        self.record = ManifestDocument(
            document_id="doc-1",
            relative_path="manual.txt",
            filename="manual.txt",
            content_hash="abcdef123456",
            chunk_ids=["chunk-1"],
            page_count=1,
            chunk_count=1,
            embedding_model="embed",
            chunk_size=700,
            chunk_overlap=100,
        )

    def setup(self):
        return None

    def manifest(self):
        return IngestionManifest(documents={} if self.deleted else {"doc-1": self.record})

    def delete_document(self, document_id: str):
        self.deleted.append(document_id)
        return True

    def index_document(self, path: Path):
        self.indexed.append(path)
        return IngestionResult(document_id="doc-1", success=True, chunk_count=1)

    def rebuild(self):
        self.rebuilt = True
        return 1

    def reconcile_index(self):
        return ReconciliationResult(orphan_chunk_ids=["orphan"])

    def chunk_count(self):
        return 1


def test_document_callbacks_delete_reindex_and_rebuild(tmp_path: Path) -> None:
    manager = FakeManager(tmp_path)
    (tmp_path / "manual.txt").write_text("changed", encoding="utf-8")
    app = RAGApplication(vector_db=manager)  # type: ignore[arg-type]

    rows, status, _, selector = app.delete_selected("doc-1")
    assert rows == []
    assert selector["choices"] == []
    assert manager.deleted == ["doc-1"]
    assert "Deleted" in status

    manager.deleted.clear()
    rows, status, errors, selector = app.reindex_changed()
    assert rows[0][0] == "manual.txt"
    assert manager.indexed == [tmp_path / "manual.txt"]
    assert "Reindexed" in status
    assert errors == []
    assert selector["choices"] == ["doc-1"]

    _, status, _, selector = app.rebuild_index()
    assert manager.rebuilt
    assert "Rebuilt" in status
    assert selector["choices"] == ["doc-1"]


def test_preflight_controls_chat_and_reports_actionable_state(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    summary, diagnostics, message_update, send_update = app.preflight(
        ollama={"reachable": False, "models": []}
    )

    assert "not ready" in summary.lower()
    assert "ollama serve" in summary
    assert message_update["interactive"] is False
    assert send_update["interactive"] is False
    assert any(row[0] == "Ollama connectivity" and row[1] == "error" for row in diagnostics)


def test_preflight_enables_chat_when_required_services_are_ready(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    summary, _, message_update, send_update = app.preflight(
        ollama={
            "reachable": True,
            "models": ["qwen3.5:9b", "nomic-embed-text:latest"],
        }
    )

    assert "ready" in summary.lower()
    assert message_update["interactive"] is True
    assert send_update["interactive"] is True


def test_preflight_requires_the_exact_configured_model_tag(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    summary, diagnostics, message_update, send_update = app.preflight(
        ollama={
            "reachable": True,
            "models": ["qwen3.5:latest", "nomic-embed-text"],
        }
    )

    assert "not ready" in summary.lower()
    assert any(row[0] == "Chat model" and row[1] == "error" for row in diagnostics)
    assert any(row[0] == "Embedding model" and row[1] == "ok" for row in diagnostics)
    assert message_update["interactive"] is False
    assert send_update["interactive"] is False


def test_export_is_public_and_trace_and_scores_preserve_observability(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    result = {
        "standalone_query": "standalone",
        "route": "complex_search",
        "strategy": "hybrid",
        "subqueries": ["one", "two"],
        "retry_count": 1,
        "evidence_status": "limited",
        "sources": [{"label": "C1", "chunk_id": "c1", "filename": "a.pdf", "page": 2}],
        "trace": [
            {
                "stage": "retrieve",
                "retrieved_count": 20,
                "fused_count": 12,
                "selected_count": 6,
                "duration_ms": 12.5,
            }
        ],
        "retrieval_hits": [
            {
                "chunk_id": "c1",
                "filename": "a.pdf",
                "page": 2,
                "semantic_score": 0.8,
                "sparse_score": 4.0,
                "fused_score": 0.03,
                "selection_score": 0.77,
                "subqueries": ["one"],
            }
        ],
        "prompt": "private",
        "reasoning": "private",
    }
    exported = app.public_export([["user", "answer"]], result)
    assert set(exported) == {
        "messages",
        "standalone_query",
        "route",
        "strategy",
        "subqueries",
        "retry_count",
        "evidence_status",
        "citations",
        "public_trace",
    }
    assert "prompt" not in json.dumps(exported)
    assert app.trace_rows(result) == [["retrieve", "", 20, 12, 6, 0, "", 12.5]]
    assert app.score_rows(result)[0][3:6] == [0.8, 4.0, 0.03]
    assert app.score_rows(result)[0][6] == 0.77
    assert app.score_rows(result)[0][-1] == "one"
    assert app.source_rows(result)[0][-1] == 0.77
    assert "Limited" in app.answer_status(result)


def test_answer_status_distinguishes_supported_abstention_and_errors(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    assert "Supported" in app.answer_status({"evidence_status": "sufficient"})
    assert "Abstention" in app.answer_status(
        {"evidence_status": "insufficient", "trace": []}
    )
    assert "Unavailable" in app.answer_status({}, error="connection refused")


def test_reconciliation_action_refreshes_document_controls(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    rows, status, errors, selector = app.reconcile_manifest_index()

    assert rows[0][0] == "manual.txt"
    assert "1 orphan" in status
    assert errors == []
    assert selector["choices"] == ["doc-1"]


def test_evaluation_tables_and_diagnostics(tmp_path: Path) -> None:
    manager = FakeManager(tmp_path)
    app = RAGApplication(vector_db=manager)  # type: ignore[arg-type]
    result = tmp_path / "run"
    result.mkdir()
    (result / "summary.json").write_text(
        json.dumps({"configuration": {"run_id": "r1"}, "metrics": {"dense": {"recall_at_5": 0.5}}}),
        encoding="utf-8",
    )
    (result / "cases.jsonl").write_text(
        json.dumps({"case_id": "x", "system": "dense", "failure_labels": ["retrieval_miss"]})
        + "\n",
        encoding="utf-8",
    )
    metrics, failures, label = app.load_evaluation_result(result)
    assert metrics[0][0:2] == ["dense", 0.5]
    assert failures[0][-1] == "retrieval_miss"
    assert "r1" in label

    diagnostics = app.diagnostic_rows(ollama={"reachable": False, "models": []})
    assert ["Orphan Chroma chunks", "warning", "1"] in diagnostics
    assert any(
        row[0] == "Embedding model" and "ollama pull nomic-embed-text" in row[2]
        for row in diagnostics
    )


def test_latest_evaluation_finds_the_newest_valid_nested_run(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    results = tmp_path / "evals" / "results"

    older = results / "multihop" / "older"
    older.mkdir(parents=True)
    (older / "summary.json").write_text("{}", encoding="utf-8")
    (older / "cases.jsonl").write_text("", encoding="utf-8")
    os.utime(older / "summary.json", (10, 10))

    newer = results / "multihop" / "newer"
    newer.mkdir(parents=True)
    (newer / "summary.json").write_text("{}", encoding="utf-8")
    (newer / "cases.jsonl").write_text("", encoding="utf-8")
    os.utime(newer / "summary.json", (20, 20))

    invalid = results / "multihop" / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "summary.json").write_text("not json", encoding="utf-8")
    (invalid / "cases.jsonl").write_text("", encoding="utf-8")
    os.utime(invalid / "summary.json", (30, 30))

    incomplete = results / "multihop" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text("{}", encoding="utf-8")
    os.utime(incomplete / "summary.json", (40, 40))

    assert RAGApplication.latest_evaluation() == newer


def test_interface_construction_does_not_require_live_ollama(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    interface = app.create_interface()
    assert interface is not None
