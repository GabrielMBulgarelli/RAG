import json
from pathlib import Path

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

    rows, status, _ = app.delete_selected("doc-1")
    assert rows == []
    assert manager.deleted == ["doc-1"]
    assert "Deleted" in status

    manager.deleted.clear()
    rows, status, errors = app.reindex_changed()
    assert rows[0][0] == "manual.txt"
    assert manager.indexed == [tmp_path / "manual.txt"]
    assert "Reindexed" in status
    assert errors == []

    _, status, _ = app.rebuild_index()
    assert manager.rebuilt
    assert "Rebuilt" in status


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
        "trace": [{"stage": "retrieve", "duration_ms": 12.5}],
        "retrieval_hits": [
            {
                "chunk_id": "c1",
                "filename": "a.pdf",
                "page": 2,
                "semantic_score": 0.8,
                "sparse_score": 4.0,
                "fused_score": 0.03,
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
    assert app.trace_rows(result) == [["retrieve", "", "", "", 0, "", 12.5]]
    assert app.score_rows(result)[0][3:6] == [0.8, 4.0, 0.03]
    assert app.score_rows(result)[0][-1] == "one"


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


def test_interface_construction_does_not_require_live_ollama(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    interface = app.create_interface()
    assert interface is not None
