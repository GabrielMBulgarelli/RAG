import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest

import modules.app as app_module
from modules.app import (
    EVALUATION_HEADERS,
    EvaluationReadiness,
    RAGApplication,
    format_duration_ms,
    format_metric,
    format_score,
    render_status,
)
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
        self.records: dict[str, ManifestDocument] = {"doc-1": self.record}

    def setup(self):
        return None

    def manifest(self):
        return IngestionManifest(documents={} if self.deleted else self.records)

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

    rows, status, _, confirmation, confirmation_update = app.delete_selected("doc-1")
    assert rows == []
    assert manager.deleted == ["doc-1"]
    assert "Deleted" in status
    assert "doc-1" not in status
    assert confirmation == ""
    assert confirmation_update["visible"] is False

    manager.deleted.clear()
    rows, status, errors = app.reindex_changed()
    assert rows[0][0] == "manual.txt"
    assert manager.indexed == [tmp_path / "manual.txt"]
    assert "Reindexed" in status
    assert errors == []

    _, status, _ = app.rebuild_index()
    assert manager.rebuilt
    assert "Rebuilt" in status


def test_document_inventory_and_selection_use_relative_path_without_exposing_id(
    tmp_path: Path,
) -> None:
    manager = FakeManager(tmp_path)
    manager.records = {
        "doc-1": manager.record,
        "doc-2": manager.record.model_copy(
            update={"document_id": "doc-2", "relative_path": "archive/manual.txt"}
        ),
    }
    app = RAGApplication(vector_db=manager)  # type: ignore[arg-type]

    rows = app.document_rows()
    assert rows == [
        ["archive/manual.txt", 1, 1, "Indexed"],
        ["manual.txt", 1, 1, "Indexed"],
    ]

    selected_id, summary, delete_button, confirmation_text, confirmation = app.select_document(
        rows, cast(Any, SimpleNamespace(index=(0, 2)))
    )

    assert selected_id == "doc-2"
    assert "archive/manual.txt" in summary["value"]
    assert "doc-2" not in summary["value"]
    assert delete_button["interactive"] is True
    assert delete_button["visible"] is True
    assert confirmation_text == ""
    assert confirmation["visible"] is False


def test_document_selection_reset_disables_delete_and_closes_confirmation(
    tmp_path: Path,
) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    selected_id, summary, delete_button, confirmation_text, confirmation = (
        app.reset_document_selection(app.document_rows())
    )

    assert selected_id == ""
    assert summary["visible"] is False
    assert delete_button["interactive"] is False
    assert delete_button["visible"] is True
    assert confirmation_text == ""
    assert confirmation["visible"] is False


def test_preflight_controls_chat_and_reports_actionable_state(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    summary, diagnostics, message_update, send_update = app.preflight(
        ollama={"reachable": False, "models": []}
    )

    assert "not ready" in summary.lower()
    assert "ollama serve" in summary
    assert message_update["interactive"] is False
    assert send_update["interactive"] is False
    assert any(row[0] == "Ollama connectivity" and row[1] == "Unavailable" for row in diagnostics)


def test_preflight_keeps_chat_disabled_until_ai_is_loaded(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    summary, _, message_update, send_update = app.preflight(
        ollama={
            "reachable": True,
            "models": ["qwen3.5:9b", "nomic-embed-text:latest"],
        }
    )

    assert "load ai models" in summary.lower()
    assert app.rag_graph is None
    assert message_update["interactive"] is False
    assert send_update["interactive"] is False


def test_manual_ai_load_initializes_graph_and_enables_chat(tmp_path: Path, monkeypatch) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    graph = object()
    monkeypatch.setattr(app_module, "RAGGraph", lambda _vector_db: graph)

    summary, diagnostics, message_update, send_update = app.load_ai_models(
        ollama={
            "reachable": True,
            "models": ["qwen3.5:9b", "nomic-embed-text:latest"],
        }
    )

    assert app.rag_graph is graph
    assert "ready for questions" in summary.lower()
    assert ["AI models", "Ready", "Loaded for this application session"] in diagnostics
    assert message_update["interactive"] is True
    assert send_update["interactive"] is True


def test_manual_ai_load_reports_missing_models_without_initializing(
    tmp_path: Path, monkeypatch
) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    initialized = False

    def initialize() -> None:
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(app, "initialize", initialize)

    summary, _, message_update, send_update = app.load_ai_models(
        ollama={"reachable": False, "models": []}
    )

    assert "not loaded" in summary.lower()
    assert initialized is False
    assert message_update["interactive"] is False
    assert send_update["interactive"] is False


def test_manual_ai_load_surfaces_initialization_failure(tmp_path: Path, monkeypatch) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    def fail_initialization() -> None:
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(app, "initialize", fail_initialization)

    summary, diagnostics, message_update, send_update = app.load_ai_models(
        ollama={
            "reachable": True,
            "models": ["qwen3.5:9b", "nomic-embed-text:latest"],
        }
    )

    assert "not loaded" in summary.lower()
    assert ["AI models", "Unavailable", "RuntimeError: index unavailable"] in diagnostics
    assert message_update["interactive"] is False
    assert send_update["interactive"] is False


def test_preflight_requires_the_exact_configured_model_tag(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    summary, diagnostics, message_update, send_update = app.preflight(
        ollama={
            "reachable": True,
            "models": ["qwen3.5:latest", "nomic-embed-text"],
        }
    )

    assert "not ready" in summary.lower()
    assert any(row[0] == "Chat model" and row[1] == "Unavailable" for row in diagnostics)
    assert any(row[0] == "Embedding model" and row[1] == "Ready" for row in diagnostics)
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
        "validation": {
            "is_valid": True,
            "violations": [],
            "repair_attempted": False,
        },
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
        "validation",
        "public_trace",
    }
    assert exported["validation"] == result["validation"]
    assert "prompt" not in json.dumps(exported)
    assert app.trace_rows(result) == [["Retrieve", "—", 20, 12, 6, 0, 0, "—", "12 ms"]]
    assert app.score_rows(result)[0][3:6] == ["0.8000", "4.0000", "0.0300"]
    assert app.score_rows(result)[0][6] == "0.7700"
    assert app.score_rows(result)[0][-1] == "one"
    assert app.source_rows(result)[0][-1] == "0.7700"
    assert "Limited" in app.answer_status(result)


def test_answer_status_distinguishes_supported_abstention_and_errors(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    assert "Supported" in app.answer_status({"evidence_status": "sufficient"})
    assert "Abstention" in app.answer_status({"evidence_status": "insufficient", "trace": []})
    assert "Unavailable" in app.answer_status({}, error="connection refused")


def test_clear_ui_rotates_the_session_checkpoint(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    cleared: list[str] = []
    app.rag_graph = cast(Any, SimpleNamespace(clear=lambda session_id: cleared.append(session_id)))

    result = app.clear_ui("session-before-clear")

    assert cleared == ["session-before-clear"]
    assert result[-1] != "session-before-clear"
    assert result[-1]


def test_reconciliation_action_refreshes_document_controls(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    rows, status, errors = app.reconcile_manifest_index()
    selected_id, summary, delete_button, confirmation_text, confirmation = (
        app.reset_document_selection(rows)
    )

    assert rows[0][0] == "manual.txt"
    assert "1 orphan" in status
    assert errors == []
    assert selected_id == ""
    assert summary["visible"] is False
    assert delete_button["visible"] is True
    assert delete_button["interactive"] is False
    assert confirmation_text == ""
    assert confirmation["visible"] is False


def test_document_deletion_requires_review_and_confirmation(tmp_path: Path) -> None:
    manager = FakeManager(tmp_path)
    app = RAGApplication(vector_db=manager)  # type: ignore[arg-type]

    confirmation, confirmation_update = app.prepare_deletion("doc-1")
    assert "manual.txt" in confirmation
    assert confirmation_update["visible"] is True
    assert manager.deleted == []

    confirmation, confirmation_update = app.cancel_deletion()
    assert confirmation == ""
    assert confirmation_update["visible"] is False
    assert manager.deleted == []

    app.delete_selected("doc-1")
    assert manager.deleted == ["doc-1"]


def test_evaluation_selection_handles_empty_and_scalar_values(tmp_path: Path, monkeypatch) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    calls: list[tuple[Path, list[str], str]] = []

    def fake_run_evaluation(
        dataset: Path, systems: list[str], split: str, *, dataset_name: str
    ) -> Path:
        calls.append((dataset, systems, split))
        assert dataset_name == "multihop"
        return tmp_path / "result"

    monkeypatch.setattr(app_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        app,
        "load_evaluation_result",
        lambda _path: ([["dense"]], [], "context", "Loaded evaluation."),
    )

    for selection in (None, [], ""):
        metrics, failures, context, status = app.run_evaluation_ui("development", selection)
        assert metrics == []
        assert failures == []
        assert context == ""
        assert "select at least one" in status.lower()
    assert calls == []

    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    dataset = tmp_path / "evals" / "multihop" / "cases.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")

    metrics, _, _, status = app.run_evaluation_ui("development", "dense")
    assert metrics == [["dense"]]
    assert "Loaded" in status
    assert calls == [(dataset, ["dense"], "development")]


def test_evaluation_readiness_represents_all_workflow_states() -> None:
    for state in ("ready", "blocked", "running", "result", "error"):
        readiness = EvaluationReadiness(state=state)
        assert readiness.state == state


def test_legacy_evaluation_summary_is_rejected(tmp_path: Path) -> None:
    manager = FakeManager(tmp_path)
    app = RAGApplication(vector_db=manager)  # type: ignore[arg-type]
    result = tmp_path / "run"
    result.mkdir()
    (result / "summary.json").write_text(
        json.dumps(
            {
                "configuration": {"run_id": "r1"},
                "metrics": {
                    "dense": {
                        "recall_at_5": 0.5,
                        "answer_token_f1": 0.25,
                        "custom_grounding_metric": 0.75,
                        "route_accuracy": 1.0,
                        "mean_latency_seconds": 99,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (result / "cases.jsonl").write_text(
        json.dumps({"case_id": "x", "system": "dense", "failure_labels": ["retrieval_miss"]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema version 2"):
        app.load_evaluation_result(result)

    diagnostics = app.diagnostic_rows(ollama={"reachable": False, "models": []})
    assert ["Orphan Chroma chunks", "warning", "1"] in diagnostics
    assert any(
        row[0] == "Embedding model" and "ollama pull nomic-embed-text" in row[2]
        for row in diagnostics
    )


def test_evaluation_loader_formats_v2_support_and_empty_statuses(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    result = tmp_path / "run-v2"
    result.mkdir()
    (result / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "configuration": {"run_id": "r2"},
                "metrics": {
                    "dense": {
                        "recall_at_5": {
                            "value": 0.0,
                            "status": "measured",
                            "sample_count": 8,
                            "note": "Cases with expected chunk evidence.",
                        },
                        "citation_precision": {
                            "value": None,
                            "status": "not_applicable",
                            "sample_count": 0,
                            "note": "Retrieval-only systems do not emit citations.",
                        },
                    },
                    "agentic": {
                        "citation_precision": {
                            "value": None,
                            "status": "no_eligible_cases",
                            "sample_count": 0,
                            "note": "No citations were emitted.",
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (result / "cases.jsonl").write_text("", encoding="utf-8")

    metrics, _, context, _ = app.load_evaluation_result(result)

    assert metrics[0][2] == "0.0% · n=8"
    citation_row = next(row for row in metrics if row[1] == "Citation precision")
    assert citation_row[2] == "— Not applicable"
    assert citation_row[5] == "— No eligible cases"
    assert "r2" in context


def test_semantic_status_banner_escapes_dynamic_content_and_exposes_aria() -> None:
    for kind in ("info", "success", "warning", "error"):
        banner = render_status(kind, f"{kind} <title>", '<script>alert("x")</script>')
        assert f"rag-status--{kind}" in banner
        expected_role = "alert" if kind == "error" else "status"
        expected_live = "assertive" if kind == "error" else "polite"
        assert f'role="{expected_role}"' in banner
        assert f'aria-live="{expected_live}"' in banner
        assert "<script>" not in banner
        assert "&lt;script&gt;" in banner


def test_result_formatters_use_consistent_precision_and_missing_values() -> None:
    assert format_metric("recall_at_5", 0.8123) == "81.2%"
    assert format_metric("p95_latency_seconds", 1.234) == "1,234 ms"
    assert format_metric("mean_llm_calls_per_query", 2.25) == "2.3"
    assert format_metric("recall_at_5", None) == "—"
    assert format_score(0.87654) == "0.8765"
    assert format_score(None) == "—"
    assert format_duration_ms(12.5) == "12 ms"
    assert format_duration_ms(None) == "—"


def test_display_rows_format_sources_traces_scores_and_diagnostics(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    result = {
        "sources": [{"label": "C1", "chunk_id": "c1", "filename": "a.pdf", "page": None}],
        "retrieval_hits": [
            {
                "chunk_id": "c1",
                "filename": "a.pdf",
                "page": None,
                "semantic_score": None,
                "sparse_score": 3,
                "fused_score": 0.02,
                "selection_score": 0.5,
                "subqueries": [],
            }
        ],
        "trace": [{"stage": "grade_evidence", "duration_ms": None}],
    }
    assert app.source_rows(result)[0][2:] == [
        "—",
        "—",
        "—",
        "3.0000",
        "0.0200",
        "0.5000",
    ]
    assert app.score_rows(result)[0][-1] == "—"
    assert app.trace_rows(result)[0][0] == "Grade evidence"
    assert app.trace_rows(result)[0][-1] == "—"

    raw = app.diagnostic_rows(ollama={"reachable": False, "models": []})
    displayed = app.diagnostic_display_rows(raw)
    assert any(row[1] == "Unavailable" for row in displayed)
    assert any(row[1] == "Review" for row in displayed)
    assert all(row[1] not in {"ok", "warning", "error", "pending"} for row in displayed)

    presentation = app.diagnostic_presentation_rows(displayed)
    assert any(row[2] == "Unavailable" for row in presentation)
    assert {row[0] for row in presentation} >= {
        "AI runtime",
        "Required models",
        "Document index",
        "Saved evaluation",
    }


def test_evidence_cards_are_accessible_and_escape_dynamic_content(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    result = {
        "sources": [
            {
                "label": "C1<script>",
                "filename": 'guide <img src="x">.pdf',
                "page": 4,
                "excerpt": "Use <b>local</b> retrieval.",
                "relevant": True,
            }
        ]
    }

    evidence = app.evidence_html(result)

    assert '<section class="evidence-list"' in evidence
    assert '<article class="evidence-item"' in evidence
    assert 'aria-label="Cited evidence"' in evidence
    assert "Cited" in evidence
    assert "Relevant" in evidence
    assert "<script>" not in evidence
    assert "<img" not in evidence
    assert "<b>" not in evidence
    assert "&lt;b&gt;local&lt;/b&gt;" in evidence
    assert "No cited evidence" in app.evidence_html({"sources": []})


def test_latest_evaluation_finds_the_newest_valid_nested_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    results = tmp_path / "evals" / "results"

    older = results / "multihop" / "older"
    older.mkdir(parents=True)
    (older / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "configuration": {
                    "dataset_name": "multihop",
                    "evaluated_split": "development",
                    "systems": ["dense", "bm25", "hybrid", "agentic"],
                },
            }
        ),
        encoding="utf-8",
    )
    (older / "cases.jsonl").write_text("", encoding="utf-8")
    os.utime(older / "summary.json", (10, 10))

    newer = results / "multihop" / "newer"
    newer.mkdir(parents=True)
    (newer / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "configuration": {
                    "dataset_name": "multihop",
                    "evaluated_split": "development",
                    "systems": ["dense", "bm25", "hybrid", "agentic"],
                },
            }
        ),
        encoding="utf-8",
    )
    (newer / "cases.jsonl").write_text("", encoding="utf-8")
    os.utime(newer / "summary.json", (20, 20))

    invalid = results / "multihop" / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "summary.json").write_text("not json", encoding="utf-8")
    (invalid / "cases.jsonl").write_text("", encoding="utf-8")
    os.utime(invalid / "summary.json", (30, 30))

    incomplete = results / "multihop" / "incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text('{"schema_version": 2}', encoding="utf-8")
    os.utime(incomplete / "summary.json", (40, 40))

    historical = results / "multihop" / "historical"
    historical.mkdir(parents=True)
    (historical / "summary.json").write_text("{}", encoding="utf-8")
    (historical / "cases.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(historical / "summary.json", (50, 50))

    unrelated = results / "custom" / "newer"
    unrelated.mkdir(parents=True)
    (unrelated / "summary.json").write_text(
        '{"schema_version": 2, "configuration": {"dataset_name": "custom"}}',
        encoding="utf-8",
    )
    (unrelated / "cases.jsonl").write_text("", encoding="utf-8")
    os.utime(unrelated / "summary.json", (60, 60))

    assert RAGApplication.latest_evaluation() == newer


def test_latest_evaluation_ignores_newer_partial_schema_v2_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "evals" / "results" / "multihop"
    standard = root / "standard"
    partial = root / "partial"
    for path, systems, modified in (
        (standard, ["dense", "bm25", "hybrid", "agentic"], 10),
        (partial, ["bm25"], 20),
    ):
        path.mkdir(parents=True)
        (path / "summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "configuration": {
                        "dataset_name": "multihop",
                        "evaluated_split": "development",
                        "systems": systems,
                    },
                }
            ),
            encoding="utf-8",
        )
        (path / "cases.jsonl").write_text("", encoding="utf-8")
        os.utime(path / "summary.json", (modified, modified))

    assert RAGApplication.latest_evaluation() == standard


def test_evaluation_context_labels_standard_and_custom_runs(tmp_path: Path) -> None:
    configuration = {
        "run_id": "run",
        "dataset_name": "multihop",
        "evaluated_split": "development",
        "timestamp": "2026-07-17T12:00:00+00:00",
    }
    standard = {
        "schema_version": 2,
        "configuration": {
            **configuration,
            "systems": ["dense", "bm25", "hybrid", "agentic"],
        },
    }
    custom = {
        "schema_version": 2,
        "configuration": {**configuration, "systems": ["bm25"]},
    }

    assert "Standard benchmark" in RAGApplication.evaluation_context_html(tmp_path, standard, 1)
    assert "Custom evaluation" in RAGApplication.evaluation_context_html(tmp_path, custom, 1)
    assert "legacy" not in RAGApplication.evaluation_context_html(tmp_path, custom, 1).lower()


def test_latest_evaluation_rejects_historical_only_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    result = tmp_path / "evals" / "results" / "multihop" / "historical"
    result.mkdir(parents=True)
    (result / "summary.json").write_text("{}", encoding="utf-8")
    (result / "cases.jsonl").write_text("{}\n", encoding="utf-8")

    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    metrics, failures, context, status = app.load_latest_evaluation()

    assert metrics == []
    assert failures == []
    assert context == ""
    assert "No standard benchmark available" in status
    assert "Run the standard benchmark" in status


def test_interface_construction_does_not_require_live_ollama(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    interface = app.create_interface()
    assert interface is not None


def test_interface_exposes_manual_ai_loading_with_automatic_workspace_refresh(
    tmp_path: Path,
) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    config = app.create_interface().get_config_file()
    labels = [component["props"].get("value") for component in config["components"]]

    assert "Load AI models" in labels
    assert "refresh_workspace_state" in json.dumps(config, default=str)


def test_interface_exposes_responsive_hierarchy_and_hides_mean_latency(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    interface = app.create_interface()
    config = interface.get_config_file()
    serialized = json.dumps(config, default=str)

    assert '"elem_id": "app-shell"' in serialized
    assert '"elem_id": "skip-navigation"' in serialized
    assert '"elem_id": "primary-tabs"' in serialized
    assert '"elem_id": "workspace-grid"' in serialized
    assert '"elem_id": "corpus-rail"' in serialized
    assert '"elem_id": "chat-workspace"' in serialized
    assert '"elem_id": "conversation-region"' in serialized
    assert '"elem_id": "evidence-list"' in serialized
    assert '"label": "Maintenance"' in serialized
    assert '"label": "Technical details"' in serialized
    assert "mean_latency_seconds" not in serialized
    assert EVALUATION_HEADERS == [
        "Category",
        "Metric",
        "Dense",
        "BM25",
        "Hybrid",
        "Agentic",
    ]

    components = [component["props"] for component in config["components"]]
    top_level_tabs = {
        props.get("elem_id"): props.get("label")
        for props in components
        if props.get("elem_id") in {"workspace-tab", "evaluation-tab", "diagnostics-tab"}
    }
    assert top_level_tabs == {
        "workspace-tab": "Workspace",
        "evaluation-tab": "Evaluation",
    }
    assert "Manage documents" not in serialized
    assert "Document ID to delete" not in serialized
    assert "Diagnostics" not in top_level_tabs.values()
    assert not any(props.get("label") in {"Documents", "Chat"} for props in components)

    action_labels = [props.get("value") for props in components]
    for label in (
        "Load AI models",
        "Index files",
        "Delete selected",
        "Cancel",
        "Confirm deletion",
        "Ask",
        "Clear",
        "Export",
        "Run standard benchmark",
    ):
        assert action_labels.count(label) == 1
    assert "Refresh document status" not in action_labels
    assert "Refresh system status" not in action_labels
    assert "Reconcile manifest/index" not in action_labels

    document_table = next(
        props for props in components if props.get("label") == "Indexed documents"
    )
    assert document_table["headers"] == ["Document", "Pages", "Chunks", "Status"]
    assert document_table["layout"] == "table"
    component_by_id = {
        component["props"].get("elem_id"): component
        for component in config["components"]
        if component["props"].get("elem_id")
    }
    dataframe_ids = {
        component["props"].get("elem_id")
        for component in config["components"]
        if component["type"] == "dataframe"
    }
    assert dataframe_ids == set()
    assert component_by_id["documents-table"]["type"] == "dataset"
    for result_id in (
        "indexing-errors-table",
        "retrieval-scores-table",
        "retrieval-trace-table",
        "evaluation-metrics-table",
        "evaluation-failures-table",
        "system-status-details",
    ):
        assert component_by_id[result_id]["type"] == "html"

    export_component = component_by_id["conversation-export"]
    assert export_component["type"] == "downloadbutton"
    assert export_component["props"]["visible"] is False


def test_local_stylesheet_covers_mobile_tables_and_keyboard_focus() -> None:
    stylesheet = Path(app_module.__file__).with_name("app.css")

    assert stylesheet.is_file()
    css = stylesheet.read_text(encoding="utf-8")
    assert "@media (max-width: 1050px)" in css
    assert "@media (max-width: 640px)" in css
    assert "color-scheme: dark" in css
    assert "overflow-x: auto" in css
    assert ":focus-visible" in css
    assert "min-height: 44px" in css
    assert "--rag-status-text:" in css
    assert "--rag-status-surface:" in css
    assert "min-width: 0" in css
    assert "overflow-x: clip" in css
    assert ".result-scroll" in css
    assert ".result-table" in css
    assert "#skip-navigation" in css
    assert "height: clamp(200px, 30vh, 320px)" in css
    assert ".stack-on-mobile" in css
    assert "--rag-info-surface:" in css
    assert "--rag-error-surface:" in css
    assert "font-family: var(--font)" in css
    assert ".evaluation-metrics-table table" in css
    assert "#workspace-grid" in css
    assert "#corpus-rail" in css
    assert ".evidence-item" in css
    assert "scrollbar-width: thin" in css
    assert "::-webkit-scrollbar" in css
    assert "8px" in css
    assert "#eef2ff" not in css
    assert "#fff1f2" not in css
    assert ".status-host > div:not(.rag-status)" in css
    assert ".status-host > div," not in css
    assert "#document-upload" in css
    assert "#evaluation-split" in css
    assert "#evaluation-systems" in css
    assert "#evaluation-setup" in css
    assert ".evaluation-config-row" in css
    assert ".evaluation-actions-row" in css
    assert ".evaluation-context" in css
    assert ".metric-value--neutral" in css
    assert ".metric-support" in css
    assert ".evaluation-toolbar" not in css
    assert "  .view-heading {\n    flex-direction: column;" in css
    config_row_rule = css.split(".evaluation-config-row {", 1)[1].split("}", 1)[0]
    assert "display: grid" not in config_row_rule
    assert "flex-wrap: wrap" in config_row_rule


def test_statuses_have_typed_aria_semantics_and_escape_content() -> None:
    informational = app_module.render_status("info", "Ready", "Use <local> models")
    failure = app_module.render_status("error", "Failed <now>", 'Bad "value"')

    assert 'role="status"' in informational
    assert 'aria-live="polite"' in informational
    assert "Use &lt;local&gt; models" in informational
    assert 'role="alert"' in failure
    assert 'aria-live="assertive"' in failure
    assert "Failed &lt;now&gt;" in failure
    assert "Bad &quot;value&quot;" in failure


def test_semantic_result_renderer_is_accessible_safe_and_handles_empty_rows() -> None:
    rendered = app_module.render_result_table(
        ["Name", "Status"],
        [["<script>alert(1)</script>", "Unavailable"]],
        caption="Readiness <checks>",
        empty_message="No checks.",
        mobile_cards=True,
    )
    empty = app_module.render_result_table(
        ["Name"],
        [],
        caption="Indexing errors",
        empty_message="No indexing errors.",
    )

    assert '<div class="result-scroll"' in rendered
    assert "<caption>Readiness &lt;checks&gt;</caption>" in rendered
    assert 'data-label="Status"' in rendered
    assert 'data-status="error"' in rendered
    assert "stack-on-mobile" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>" not in rendered
    assert 'class="result-empty"' in empty
    assert "No indexing errors." in empty


def test_result_row_normalizer_accepts_only_tabular_sequences() -> None:
    assert app_module.normalize_result_rows([["one", 2], ("three", 4)]) == [
        ["one", 2],
        ["three", 4],
    ]
    assert app_module.normalize_result_rows({"unexpected": "mapping"}) == []
    assert app_module.normalize_result_rows("unexpected text") == []


def test_dynamic_statuses_are_accessible_live_regions(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    config = app.create_interface().get_config_file()
    components = [component["props"] for component in config["components"]]

    live_regions = {
        props.get("elem_id"): props
        for props in components
        if props.get("elem_id", "").endswith("-status")
    }
    assert {
        "readiness-status",
        "ingestion-status",
        "answer-status",
        "evaluation-status",
    } <= live_regions.keys()
    assert live_regions["readiness-status"].get("value", "").count('role="status"') == 1
    for idle_id in ("ingestion-status", "answer-status", "evaluation-status"):
        assert live_regions[idle_id].get("value", "") == ""
        assert live_regions[idle_id].get("visible") is False
    assert "requestAnimationFrame" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "addedNodes" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "ResizeObserver" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "horizontally scrollable" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "vertically scrollable" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "dataset.overflowX" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "dataset.overflowY" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "target.dataset.overflowX !== nextOverflowX" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "target.dataset.overflowY !== nextOverflowY" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'querySelector(".result-scroll, .wrap")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'target.setAttribute("role", "region")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'target.removeAttribute("tabindex")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'target.removeAttribute("aria-label")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "resizeObserver.observe(target)" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'querySelector("button.label-wrap")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'classList.contains("open")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'setAttribute("aria-expanded"' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert '"corpus-rail":' not in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'type="range"' not in app_module.ACCESSIBILITY_BOOTSTRAP


def test_system_status_groups_problems_and_preserves_technical_values(
    tmp_path: Path,
) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    rows = [
        ["AI runtime", "Ollama connectivity", "Unavailable", "Run ollama serve"],
        ["Document index", "Manifest", "Ready", "Valid; 1 document"],
    ]

    rendered = app.system_status_html(rows)

    assert 'aria-label="System status"' in rendered
    assert "Action required" in rendered
    assert "Ollama connectivity" in rendered
    assert "Run ollama serve" in rendered
    assert "Technical values" in rendered
    assert "Valid; 1 document" in rendered


def test_system_status_never_treats_unchecked_state_as_ready(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    rendered = app.system_status_html([])

    assert "Status unknown" in rendered
    assert "System ready" not in rendered
    assert "<details" not in rendered


def test_document_samples_are_filtered_without_exposing_ids(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    assert app.document_samples("MANUAL") == [["manual.txt", 1, 1, "Indexed"]]
    assert "doc-1" not in json.dumps(app.document_samples(""))


def test_evaluation_adapters_reveal_results_and_status(tmp_path: Path, monkeypatch) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    monkeypatch.setattr(
        app,
        "load_latest_evaluation",
        lambda: (
            [["Retrieval", "Recall at 5", "50.0%", "—", "—", "—"]],
            [],
            '<section aria-label="Evaluation result context">r1</section>',
            "loaded",
        ),
    )

    (
        results_panel,
        context,
        metrics,
        failures,
        failure_panel,
        status,
        run_button,
        load_button,
    ) = app.load_latest_evaluation_ui()

    assert results_panel["visible"] is True
    assert context["visible"] is True
    assert "r1" in context["value"]
    assert metrics["visible"] is True
    assert "Recall at 5" in metrics["value"]
    assert "No evaluation failures" in failures
    assert failure_panel["visible"] is True
    assert status["visible"] is True
    assert run_button["interactive"] is True
    assert load_button["interactive"] is True


def test_evaluation_adapters_hide_results_on_validation_error(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    updates = app.run_evaluation_presentation_ui("development", [])

    assert updates[0]["visible"] is False
    assert updates[1]["visible"] is False
    assert updates[2]["visible"] is False
    assert updates[4]["visible"] is False
    assert "select at least one" in updates[5]["value"].lower()
    assert updates[6]["interactive"] is False
    assert updates[7]["interactive"] is True


def test_evaluation_running_state_disables_both_actions(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    run_button, load_button, status = app.begin_evaluation_ui()

    assert run_button["interactive"] is False
    assert load_button["interactive"] is False
    assert status["visible"] is True
    assert "Running evaluation" in status["value"]


def test_evaluation_context_escapes_metadata(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    result = tmp_path / "unsafe-run"
    result.mkdir()
    (result / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "configuration": {
                    "run_id": "<script>bad()</script>",
                    "dataset_name": "multi<hop",
                    "evaluated_split": "development",
                    "systems": ["dense", "agentic"],
                    "timestamp": "2026-07-17T12:00:00+00:00",
                },
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    (result / "cases.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    _, _, context, _ = app.load_evaluation_result(result)

    assert "<script>" not in context
    assert "&lt;script&gt;" in context
    assert "multi&lt;hop" in context
    assert "Dense, Agentic" in context
    assert "2 cases" in context
    assert "2026-07-17 12:00 UTC" in context


def test_evaluation_interface_has_task_order_and_four_systems(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    config = app.create_interface().get_config_file()
    components = [component["props"] for component in config["components"]]
    by_id = {props.get("elem_id"): props for props in components if props.get("elem_id")}

    assert by_id["evaluation-split"]["label"] == "Split"
    assert by_id["evaluation-split"]["choices"] == [
        ("Development", "development"),
        ("Test — held-out final validation", "test"),
    ]
    assert by_id["evaluation-systems"]["label"] == "Systems"
    assert by_id["evaluation-systems"]["choices"] == [
        ("dense", "dense"),
        ("bm25", "bm25"),
        ("hybrid", "hybrid"),
        ("agentic", "agentic"),
    ]
    assert by_id["evaluation-systems"]["value"] == list(app_module.SYSTEMS)
    assert by_id["evaluation-advanced-options"]["open"] is False
    assert by_id["evaluation-results"]["visible"] is False
    assert by_id["evaluation-result-context"]["visible"] is False
    action_values = [props.get("value") for props in components]
    assert action_values.count("Run standard benchmark") == 1
    assert action_values.count("Refresh latest result") == 1
    assert "all" not in by_id["evaluation-systems"]["value"]

    component_ids = {
        component["props"].get("elem_id"): component["id"]
        for component in config["components"]
        if component["props"].get("elem_id")
    }
    parents: dict[int, int] = {}

    def record_parents(node: Mapping[str, Any], parent_id: int | None = None) -> None:
        node_id = node.get("id")
        if node_id is not None and parent_id is not None:
            parents[node_id] = parent_id
        for child in node.get("children", []):
            record_parents(child, node_id)

    layout = config.get("layout")
    assert layout is not None
    record_parents(cast(Mapping[str, Any], layout))
    run_parent = parents[component_ids["run-evaluation"]]
    assert parents[component_ids["load-evaluation"]] == run_parent
    assert parents[component_ids["evaluation-status"]] != run_parent
