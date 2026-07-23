import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import gradio as gr

from modules.app import RAGApplication
from modules.models import (
    IngestionManifest,
    IngestionResult,
    ManifestDocument,
    ReconciliationResult,
)


class ParitySettings:
    def __init__(self, sources_dir: Path) -> None:
        self.sources_dir = sources_dir


class ParityManager:
    def __init__(self, root: Path) -> None:
        self.settings = ParitySettings(root)
        self.records = {
            "doc-pdf": ManifestDocument(
                document_id="doc-pdf",
                relative_path="guide.pdf",
                filename="guide.pdf",
                content_hash="unchanged",
                chunk_ids=["pdf-1"],
                page_count=3,
                chunk_count=1,
                embedding_model="embed",
                chunk_size=700,
                chunk_overlap=100,
            ),
            "doc-txt": ManifestDocument(
                document_id="doc-txt",
                relative_path="notes.txt",
                filename="notes.txt",
                content_hash="unchanged",
                chunk_ids=["txt-1"],
                page_count=1,
                chunk_count=1,
                embedding_model="embed",
                chunk_size=700,
                chunk_overlap=100,
            ),
        }
        self.deleted: list[str] = []
        self.indexed: list[Path] = []

    def setup(self) -> None:
        return None

    def save_uploads(self, paths: list[str] | None) -> list[Path]:
        return [Path(path) for path in paths or []]

    def manifest(self) -> IngestionManifest:
        return IngestionManifest(
            documents={key: value for key, value in self.records.items() if key not in self.deleted}
        )

    def index_document(self, path: str | Path) -> IngestionResult:
        normalized = Path(path)
        self.indexed.append(normalized)
        return IngestionResult(
            document_id=f"indexed-{normalized.name}",
            success=True,
            chunk_count=1,
        )

    def has_deleted_document(self, document_id: str) -> bool:
        self.deleted.append(document_id)
        return document_id in self.records

    delete_document = has_deleted_document

    def rebuild(self) -> int:
        return 2

    def reconcile_index(self) -> ReconciliationResult:
        return ReconciliationResult()

    def chunk_count(self) -> int:
        return 2


class ParityGraph:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.sessions: list[str] = []
        self.cleared: list[str] = []

    def process_query(self, _message: str, session_id: str) -> dict[str, Any]:
        self.sessions.append(session_id)
        return dict(self.result)

    def clear(self, session_id: str) -> None:
        self.cleared.append(session_id)


def test_runtime_remains_manual_and_actionable_without_ollama(tmp_path: Path) -> None:
    # Arrange
    application = RAGApplication(vector_db=ParityManager(tmp_path))

    # Act
    from modules.ui.shell import build_application

    interface = build_application(application)
    summary, diagnostics, message_update, send_update = application.preflight(
        ollama={"reachable": False, "models": []}
    )

    # Then
    assert isinstance(interface, gr.Blocks)
    assert application.rag_graph is None
    assert "ollama serve" in summary
    assert any("ollama pull qwen3.5:9b" in row[2] for row in diagnostics)
    assert any("ollama pull nomic-embed-text" in row[2] for row in diagnostics)
    assert message_update["interactive"] is False
    assert send_update["interactive"] is False


def test_controller_no_longer_exposes_the_legacy_page_builder() -> None:
    assert not hasattr(RAGApplication, "create_interface")
    assert not hasattr(RAGApplication, "create_legacy_interface")


def test_document_lifecycle_data_and_confirmation_remain_available(tmp_path: Path) -> None:
    # Arrange
    manager = ParityManager(tmp_path)
    application = RAGApplication(vector_db=manager)
    pdf = tmp_path / "guide.pdf"
    text = tmp_path / "notes.txt"

    # Act
    rows, status, errors, _ = application.index_selected([str(pdf), str(text)])
    selected_id, summary, delete_button, _, _ = application.select_document(
        rows, cast(Any, SimpleNamespace(index=(1, 0)))
    )
    confirmation, confirmation_update = application.prepare_deletion(selected_id)

    # Then
    assert manager.indexed == [pdf, text]
    assert rows == [["guide.pdf", 3, 1, "Indexed"], ["notes.txt", 1, 1, "Indexed"]]
    assert all("doc-" not in str(row) for row in rows)
    assert application.document_samples("NOTES") == [["notes.txt", 1, 1, "Indexed"]]
    assert "2 document(s) indexed" in status
    assert errors == []
    assert "notes.txt" in summary["value"]
    assert delete_button["interactive"] is True
    assert "Delete notes.txt" in confirmation
    assert confirmation_update["visible"] is True


def test_conversation_observability_export_and_session_reset_are_preserved(
    tmp_path: Path,
) -> None:
    # Arrange
    application = RAGApplication(vector_db=ParityManager(tmp_path))
    result = {
        "answer": "The guide recommends local retrieval [C1].",
        "standalone_query": "What does the guide recommend?",
        "route": "complex_search",
        "strategy": "hybrid",
        "subqueries": ["guide recommendation", "local retrieval"],
        "retry_count": 1,
        "evidence_status": "limited",
        "conflict": False,
        "sources": [
            {
                "label": "C1",
                "chunk_id": "pdf-1",
                "filename": "guide.pdf",
                "page": 2,
                "excerpt": "Use local retrieval.",
            }
        ],
        "retrieval_hits": [
            {
                "chunk_id": "pdf-1",
                "filename": "guide.pdf",
                "page": 2,
                "semantic_score": 0.8,
                "sparse_score": 1.2,
                "fused_score": 0.7,
                "selection_score": 0.9,
                "subqueries": ["guide recommendation"],
            }
        ],
        "trace": [
            {
                "stage": "retrieve",
                "decision": "hybrid",
                "retrieved_count": 4,
                "fused_count": 3,
                "selected_count": 1,
                "retry_count": 1,
                "llm_calls": 1,
                "termination": "limited",
                "duration_ms": 12.0,
            }
        ],
        "validation": {"is_valid": True, "violations": [], "repair_attempted": False},
        "private_prompt": "must not leave the controller",
        "reasoning": "must not leave the controller",
    }
    graph = ParityGraph(result)
    application.rag_graph = cast(Any, graph)

    # Act
    _, messages, public_result, answer_state, diagnostics, evidence, scores, trace = (
        application.chat("What does the guide recommend?", [], "session-a")
    )
    exported = application.public_export(messages, public_result)
    cleared = application.clear("session-a")

    # Then
    assert graph.sessions == ["session-a"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "Limited" in answer_state
    assert all(value in diagnostics for value in ("complex_search", "hybrid", "Retries: **1**"))
    assert all(value in evidence for value in ("C1", "guide.pdf", "Use local retrieval."))
    assert scores[0][3:7] == ["0.8000", "1.2000", "0.7000", "0.9000"]
    assert trace[0][0:8] == ["Retrieve", "Hybrid", 4, 3, 1, 1, 1, "Limited"]
    assert exported["validation"]["is_valid"] is True
    assert exported["public_trace"] == result["trace"]
    assert "private_prompt" not in exported
    assert "reasoning" not in exported
    assert graph.cleared == ["session-a"]
    assert "Conversation cleared" in cleared[2]


def test_answer_states_and_recoverable_failure_remain_distinct(tmp_path: Path) -> None:
    # Arrange
    application = RAGApplication(vector_db=ParityManager(tmp_path))

    # Then
    assert "Supported" in application.answer_status({"evidence_status": "sufficient"})
    assert "Limited" in application.answer_status({"evidence_status": "limited"})
    assert "Abstention" in application.answer_status({"evidence_status": "insufficient"})
    assert "Unavailable" in application.answer_status({}, error="Ollama stopped")
    assert "Completed" in application.answer_status({})

    application.rag_graph = cast(
        Any,
        SimpleNamespace(
            process_query=lambda _message, _session: (_ for _ in ()).throw(
                RuntimeError("temporary failure")
            )
        ),
    )
    previous = [{"role": "assistant", "content": "Earlier answer"}]

    # Act
    _, messages, _, state, diagnostics, *_ = application.chat("Retry this", previous, "session-b")

    # Then
    assert messages[:1] == previous
    assert [message["role"] for message in messages[1:]] == ["user", "assistant"]
    assert "Unavailable" in state
    assert "temporary failure" in diagnostics


def test_schema_v2_evaluation_keeps_classification_metrics_and_failures(tmp_path: Path) -> None:
    # Arrange
    result_dir = tmp_path / "standard-run"
    result_dir.mkdir()
    summary = {
        "schema_version": 2,
        "configuration": {
            "run_id": "standard-run",
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
                    "note": "Cases with expected evidence.",
                },
                "citation_precision": {
                    "value": None,
                    "status": "not_applicable",
                    "sample_count": 0,
                    "note": "Retrieval-only systems do not cite.",
                },
            }
        },
    }
    (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (result_dir / "cases.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "system": "dense",
                "route": "simple_search",
                "strategy": "semantic",
                "failure_labels": ["retrieval_miss"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Act
    metrics, failures, context, status = RAGApplication.load_evaluation_result(result_dir)
    recall = next(row for row in metrics if row[1] == "Recall at 5")
    citation = next(row for row in metrics if row[1] == "Citation precision")

    # Then
    assert recall[2] == "75.0% · n=8"
    assert citation[2] == "— Not applicable"
    assert failures == [["case-1", "Dense", "Simple search", "Semantic", "Retrieval miss"]]
    assert "Standard benchmark" in context
    assert "development" in context
    assert "Standard benchmark loaded" in status
