import json
import os
from pathlib import Path

import modules.app as app_module
from modules.app import (
    DISPLAY_METRIC_LABELS,
    EVALUATION_HEADERS,
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

    rows, status, _, selector, confirmation, confirmation_update = app.delete_selected("doc-1")
    assert rows == []
    assert selector["choices"] == []
    assert manager.deleted == ["doc-1"]
    assert "Deleted" in status
    assert confirmation == ""
    assert confirmation_update["visible"] is False

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
    assert any(
        row[0] == "Ollama connectivity" and row[1] == "Unavailable"
        for row in diagnostics
    )


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
    assert app.trace_rows(result) == [
        ["Retrieve", "—", 20, 12, 6, 0, "—", "12 ms"]
    ]
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


def test_reconciliation_action_refreshes_document_controls(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]

    rows, status, errors, selector = app.reconcile_manifest_index()

    assert rows[0][0] == "manual.txt"
    assert "1 orphan" in status
    assert errors == []
    assert selector["choices"] == ["doc-1"]


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

    def fake_run_evaluation(dataset: Path, systems: list[str], split: str) -> Path:
        calls.append((dataset, systems, split))
        return tmp_path / "result"

    monkeypatch.setattr(app_module, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(
        app,
        "load_evaluation_result",
        lambda _path: ([["dense"]], [], "Loaded evaluation."),
    )

    for selection in (None, [], ""):
        metrics, failures, status = app.run_evaluation_ui("development", selection)
        assert metrics == []
        assert failures == []
        assert "select at least one" in status.lower()
    assert calls == []

    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    dataset = tmp_path / "evals" / "mvp_cases.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}\n", encoding="utf-8")

    metrics, _, status = app.run_evaluation_ui("development", "dense")
    assert metrics == [["dense"]]
    assert "Loaded" in status
    assert calls == [(dataset, ["dense"], "development")]


def test_evaluation_tables_and_diagnostics(tmp_path: Path) -> None:
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
    metrics, failures, label = app.load_evaluation_result(result)
    assert metrics[0] == ["Retrieval", "Recall at 5", "50.0%", "—", "—", "—"]
    assert ["Answer quality", "Answer token F1", "25.0%", "—", "—", "—"] in metrics
    assert metrics[-1] == [
        "Other",
        "Custom grounding metric",
        "75.0%",
        "—",
        "—",
        "—",
    ]
    assert not any(row[1] in {"Route accuracy", "Mean latency"} for row in metrics)
    assert len(metrics) == len(DISPLAY_METRIC_LABELS) + 1
    assert failures[0][-1] == "Retrieval miss"
    assert "r1" in label

    diagnostics = app.diagnostic_rows(ollama={"reachable": False, "models": []})
    assert ["Orphan Chroma chunks", "warning", "1"] in diagnostics
    assert any(
        row[0] == "Embedding model" and "ollama pull nomic-embed-text" in row[2]
        for row in diagnostics
    )


def test_semantic_status_banner_escapes_dynamic_content_and_exposes_aria() -> None:
    for kind in ("info", "success", "warning", "error"):
        banner = render_status(kind, f"{kind} <title>", '<script>alert("x")</script>')
        assert f"rag-status--{kind}" in banner
        assert 'role="status"' in banner
        assert 'aria-live="polite"' in banner
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
        "sources": [
            {"label": "C1", "chunk_id": "c1", "filename": "a.pdf", "page": None}
        ],
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


def test_interface_exposes_manual_ai_loading_without_automatic_preflight(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    config = app.create_interface().get_config_file()
    labels = [component["props"].get("value") for component in config["components"]]

    assert "Load AI models" in labels
    assert not any(
        dependency.get("trigger_after") == "load"
        for dependency in config.get("dependencies", [])
    )


def test_interface_exposes_responsive_hierarchy_and_hides_mean_latency(tmp_path: Path) -> None:
    app = RAGApplication(vector_db=FakeManager(tmp_path))  # type: ignore[arg-type]
    interface = app.create_interface()
    config = interface.get_config_file()
    serialized = json.dumps(config, default=str)

    assert '"elem_id": "app-shell"' in serialized
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
        "diagnostics-tab": "Diagnostics",
    }
    assert not any(props.get("label") in {"Documents", "Chat"} for props in components)

    action_labels = [props.get("value") for props in components]
    for label in (
        "Load AI models",
        "Index document",
        "Review deletion",
        "Cancel",
        "Confirm deletion",
        "Ask",
        "Clear",
        "Export",
        "Run evaluation",
        "Refresh diagnostics",
    ):
        assert action_labels.count(label) == 1

    document_table = next(
        props for props in components if props.get("label") == "Indexed documents"
    )
    assert document_table["show_search"] == "filter"
    assert document_table["wrap"] is True
    assert document_table["max_height"] == 360
    table_ids = {
        props.get("elem_id")
        for props in components
        if "rag-table" in (props.get("elem_classes") or [])
    }
    assert {
        "documents-table",
        "indexing-errors-table",
        "retrieval-scores-table",
        "retrieval-trace-table",
        "evaluation-metrics-table",
        "evaluation-failures-table",
        "diagnostics-table",
    } <= table_ids


def test_local_stylesheet_covers_mobile_tables_and_keyboard_focus() -> None:
    stylesheet = Path(app_module.__file__).with_name("app.css")

    assert stylesheet.is_file()
    css = stylesheet.read_text(encoding="utf-8")
    assert "@media (max-width: 900px)" in css
    assert "@media (max-width: 640px)" in css
    assert "overflow-x: auto" in css
    assert ":focus-visible" in css
    assert "min-height: 44px" in css
    assert "--rag-status-text:" in css
    assert "--rag-status-surface:" in css
    assert "min-width: 0" in css
    assert "overflow-x: clip" in css
    assert ".rag-table > div" in css
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
    assert all(props.get("value", "").count('role="status"') == 1 for props in live_regions.values())
    assert 'setAttribute("role", "alert")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "requestAnimationFrame" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "addedNodes" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "ResizeObserver" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "horizontally scrollable" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "vertically scrollable" in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'dataset.overflowX' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'dataset.overflowY' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'region.id !== "corpus-rail"' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'removeAttribute("tabindex")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'removeAttribute("aria-label")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'querySelector("button.label-wrap")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'classList.contains("open")' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert 'setAttribute("aria-expanded"' in app_module.ACCESSIBILITY_BOOTSTRAP
    assert "for (const node of pending) enhanceNode(node);\n      syncCorpusMode();" in (
        app_module.ACCESSIBILITY_BOOTSTRAP
    )
    assert 'type="range"' not in app_module.ACCESSIBILITY_BOOTSTRAP
