"""Gradio interface exposing the existing local RAG services."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from html import escape
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import gradio as gr
from gradio.themes import Soft

from modules.config import PROJECT_ROOT, config
from modules.evaluation import SYSTEMS, run_evaluation
from modules.rag_graph import RAGGraph
from modules.vector_db import VectorDBManager

DOCUMENT_HEADERS = [
    "Filename",
    "Source path",
    "Document ID",
    "Pages",
    "Chunks",
    "Content hash",
    "Status",
    "Last error",
]
TRACE_HEADERS = [
    "Stage",
    "Decision",
    "Retrieved",
    "Fused",
    "Selected",
    "Retry",
    "Termination",
    "Duration (ms)",
]
SCORE_HEADERS = [
    "Chunk ID",
    "Filename",
    "Page",
    "Semantic",
    "Sparse",
    "Fused",
    "Selection",
    "Matched subqueries",
]
METRIC_NAMES = [
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "route_accuracy",
    "strategy_accuracy",
    "retry_precision",
    "retry_recall",
    "citation_precision",
    "gold_evidence_citation_coverage",
    "abstention_accuracy",
    "conflict_accuracy",
    "termination_rate",
    "mean_latency_seconds",
    "p95_latency_seconds",
    "mean_llm_calls_per_query",
    "mean_retrieval_rounds_per_query",
]
DISPLAY_METRIC_NAMES = [name for name in METRIC_NAMES if name != "mean_latency_seconds"]
DISPLAY_METRIC_LABELS = {
    "recall_at_5": "Recall at 5",
    "mrr_at_5": "MRR at 5",
    "ndcg_at_5": "NDCG at 5",
    "route_accuracy": "Route accuracy",
    "strategy_accuracy": "Strategy accuracy",
    "retry_precision": "Retry precision",
    "retry_recall": "Retry recall",
    "citation_precision": "Citation precision",
    "gold_evidence_citation_coverage": "Gold evidence citation coverage",
    "abstention_accuracy": "Abstention accuracy",
    "conflict_accuracy": "Conflict accuracy",
    "termination_rate": "Termination rate",
    "p95_latency_seconds": "P95 latency",
    "mean_llm_calls_per_query": "Mean LLM calls per query",
    "mean_retrieval_rounds_per_query": "Mean retrieval rounds per query",
}
EVALUATION_SYSTEMS = ("dense", "bm25", "hybrid", "agentic")
EVALUATION_HEADERS = ["Metric", "Dense", "BM25", "Hybrid", "Agentic"]
PERCENTAGE_METRICS = {
    "recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "route_accuracy",
    "strategy_accuracy",
    "retry_precision",
    "retry_recall",
    "citation_precision",
    "gold_evidence_citation_coverage",
    "abstention_accuracy",
    "conflict_accuracy",
    "termination_rate",
}
StatusKind = Literal["info", "success", "warning", "error"]
APP_STYLESHEET = Path(__file__).with_name("app.css")


def readable_label(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("_", " ").strip().capitalize()


def format_score(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.4f}"


def format_duration_ms(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{round(float(value)):,} ms"


def format_metric(name: str, value: Any) -> str:
    if value is None or value == "":
        return "—"
    if name in PERCENTAGE_METRICS:
        return f"{float(value) * 100:.1f}%"
    if name == "p95_latency_seconds":
        return format_duration_ms(float(value) * 1000)
    rounded = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded:.1f}"


def render_status(kind: StatusKind, title: str, detail: str = "") -> str:
    icons = {"info": "i", "success": "✓", "warning": "!", "error": "×"}
    safe_title = escape(title)
    safe_detail = escape(detail)
    detail_html = f'<span class="rag-status__detail">{safe_detail}</span>' if detail else ""
    return (
        f'<div class="rag-status rag-status--{kind}" role="status" '
        'aria-live="polite" aria-atomic="true">'
        f'<span class="rag-status__icon" aria-hidden="true">{icons[kind]}</span>'
        '<span class="rag-status__copy">'
        f'<strong class="rag-status__title">{safe_title}</strong>{detail_html}'
        "</span></div>"
    )


ACCESSIBILITY_BOOTSTRAP = """
() => {
  const tableLabels = {
    "documents-table": "Indexed documents",
    "indexing-errors-table": "Indexing errors",
    "sources-table": "Sources used in this answer",
    "retrieval-scores-table": "Retrieval scores",
    "retrieval-trace-table": "Retrieval trace",
    "evaluation-metrics-table": "Evaluation metrics comparison",
    "evaluation-failures-table": "Evaluation failure cases",
    "diagnostics-table": "Readiness checks",
  };
  const syncTable = (table) => {
    const label = tableLabels[table.id] || "Data table";
    const grid = table.querySelector("table");
    const scrollable = Boolean(grid && grid.scrollWidth > table.clientWidth + 1);
    table.setAttribute("role", "region");
    table.setAttribute("aria-label", scrollable ? `${label}, horizontally scrollable` : label);
    table.toggleAttribute("tabindex", scrollable);
    table.dataset.scrollable = String(scrollable);
    table.dataset.empty = String(Boolean(grid && grid.tBodies[0]?.rows.length === 0));
  };
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) syncTable(entry.target);
  });
  const enhanceNode = (node) => {
    if (!(node instanceof Element)) return;
    const deletion = document.getElementById("deletion-alert");
    if (deletion) {
      deletion.setAttribute("role", "alert");
      deletion.setAttribute("aria-live", "assertive");
    }
    const tables = node.matches(".rag-table")
      ? [node]
      : Array.from(node.querySelectorAll(".rag-table"));
    for (const table of tables) {
      syncTable(table);
      if (!table.dataset.resizeObserved) {
        table.dataset.resizeObserved = "true";
        resizeObserver.observe(table);
      }
    }
  };
  enhanceNode(document.body);
  const pending = new Set();
  let frame = null;
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) pending.add(node);
    }
    if (frame !== null) return;
    frame = requestAnimationFrame(() => {
      for (const node of pending) enhanceNode(node);
      pending.clear();
      frame = null;
    });
  }).observe(document.body, {
    childList: true,
    subtree: true,
  });
}
"""


class RAGApplication:
    def __init__(self, vector_db: VectorDBManager | None = None):
        self.vector_db = vector_db or VectorDBManager()
        self.rag_graph: RAGGraph | None = None
        self.last_errors: dict[str, str] = {}

    def initialize(self) -> None:
        self.vector_db.setup()
        self.rag_graph = RAGGraph(self.vector_db)

    def _graph(self) -> RAGGraph:
        if self.rag_graph is None:
            self.initialize()
        assert self.rag_graph is not None
        return self.rag_graph

    def document_rows(self) -> list[list[Any]]:
        manifest = self.vector_db.manifest()
        return [
            [
                record.filename,
                record.relative_path,
                record.document_id,
                record.page_count,
                record.chunk_count,
                record.content_hash[:12],
                "Indexed",
                self.last_errors.get(record.document_id, "—"),
            ]
            for record in sorted(manifest.documents.values(), key=lambda item: item.relative_path)
        ]

    def document_selector_update(self) -> dict[str, Any]:
        choices = [row[2] for row in self.document_rows()]
        return gr.update(choices=choices, value=None)

    @staticmethod
    def error_rows(errors: list[Any]) -> list[list[str]]:
        return [
            [
                error.document or "—",
                readable_label(error.operation),
                readable_label(error.error_type),
                error.message or "—",
            ]
            for error in errors
        ]

    def _reset_graph(self) -> None:
        self.rag_graph = None

    def index_selected(
        self, files: list[str] | None, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        progress(0, desc="Saving files")
        paths = self.vector_db.save_uploads(files or [])
        errors = []
        total_chunks = 0
        for index, path in enumerate(paths):
            progress((index, max(len(paths), 1)), desc=f"Parsing and chunking {path.name}")
            progress(None, desc=f"Embedding and upserting {path.name}")
            result = self.vector_db.index_document(path)
            if result.success:
                total_chunks += result.chunk_count
                self.last_errors.pop(result.document_id, None)
            elif result.error:
                errors.append(result.error)
                self.last_errors[result.document_id] = result.error.message
        progress(None, desc="Removing stale chunks and updating manifest")
        self._reset_graph()
        return (
            self.document_rows(),
            render_status(
                "warning" if errors else "success",
                "Indexing completed" if not errors else "Indexing completed with errors",
                f"{len(paths) - len(errors)} document(s) indexed · {total_chunks} chunks",
            ),
            self.error_rows(errors),
            self.readiness(),
            self.document_selector_update(),
        )

    def reindex_changed(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        manifest = self.vector_db.manifest()
        changed: list[Path] = []
        for record in manifest.documents.values():
            path = self.vector_db.settings.sources_dir / record.relative_path
            if (
                path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest() != record.content_hash
            ):
                changed.append(path)
        errors = []
        for index, path in enumerate(changed):
            progress(
                (index, max(len(changed), 1)),
                desc=f"Parsing, chunking, embedding and upserting {path.name}",
            )
            result = self.vector_db.index_document(path)
            if result.error:
                errors.append(result.error)
                self.last_errors[result.document_id] = result.error.message
        progress(None, desc="Removing stale chunks and updating manifest")
        self._reset_graph()
        return (
            self.document_rows(),
            render_status(
                "warning" if errors else "success",
                "Reindexed documents" if not errors else "Reindexed with errors",
                f"{len(changed) - len(errors)} changed document(s) reindexed",
            ),
            self.error_rows(errors),
            self.document_selector_update(),
        )

    def delete_selected(self, document_id: str | None):
        if not document_id:
            return (
                self.document_rows(),
                render_status("warning", "No document selected", "Select a document to delete."),
                [],
                self.document_selector_update(),
                "",
                gr.update(visible=False),
            )
        deleted = self.vector_db.delete_document(document_id)
        self._reset_graph()
        status = render_status(
            "success" if deleted else "warning",
            "Deleted document" if deleted else "Document not found",
            document_id,
        )
        return (
            self.document_rows(),
            status,
            [],
            self.document_selector_update(),
            "",
            gr.update(visible=False),
        )

    def prepare_deletion(self, document_id: str | None):
        if not document_id:
            return "", gr.update(visible=False)
        record = self.vector_db.manifest().documents.get(document_id)
        filename = record.filename if record else document_id
        return (
            f"<strong>Delete {escape(filename)}</strong> "
            f"<code>{escape(document_id)}</code> and its indexed chunks?",
            gr.update(visible=True),
        )

    @staticmethod
    def cancel_deletion():
        return "", gr.update(visible=False)

    def rebuild_index(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        progress(None, desc="Rebuilding collection, parsing and chunking documents")
        count = self.vector_db.rebuild()
        progress(None, desc="Embedding, upserting and updating manifest")
        self._reset_graph()
        return (
            self.document_rows(),
            render_status("success", "Rebuilt complete index", f"{count} chunks available"),
            [],
            self.document_selector_update(),
        )

    def reconcile_manifest_index(self):
        try:
            result = self.vector_db.reconcile_index()
            detail = (
                f"{len(result.missing_chunk_ids)} missing, "
                f"{len(result.orphan_chunk_ids)} orphan, "
                f"{len(result.duplicate_chunk_ids)} duplicate, "
                f"{len(result.missing_source_files)} missing source, and "
                f"{len(result.incompatible_document_ids)} incompatible document ID(s)."
            )
            needs_review = any(
                (
                    result.missing_chunk_ids,
                    result.orphan_chunk_ids,
                    result.duplicate_chunk_ids,
                    result.missing_source_files,
                    result.incompatible_document_ids,
                )
            )
            status = render_status(
                "warning" if needs_review else "success",
                "Reconciliation needs review" if needs_review else "Index is reconciled",
                detail,
            )
        except Exception as exc:
            status = render_status(
                "error", "Reconciliation failed", f"{type(exc).__name__}: {exc}"
            )
        return self.document_rows(), status, [], self.document_selector_update()

    def refresh_documents(self):
        return self.document_rows(), self.readiness(), self.document_selector_update()

    def _ollama_info(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                f"{config.ollama_base_url}/api/tags", timeout=2
            ) as response:  # noqa: S310
                payload = json.load(response)
            return {
                "reachable": True,
                "models": [item.get("name", "") for item in payload.get("models", [])],
            }
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return {"reachable": False, "models": []}

    def readiness(self) -> str:
        info = self._ollama_info()
        try:
            chunks = self.vector_db.chunk_count()
            documents = len(self.vector_db.manifest().documents)
        except Exception:
            chunks, documents = 0, 0
        return render_status(
            "success" if info["reachable"] else "info",
            "Local workspace ready" if info["reachable"] else "Interface ready without AI",
            f"Ollama {'available' if info['reachable'] else 'not connected'} · "
            f"{chunks} chunks · {documents} documents",
        )

    def preflight(self, ollama: dict[str, Any] | None = None):
        rows = self.diagnostic_rows(ollama=ollama)
        blocking = [row for row in rows if row[1] == "error"]
        ready = not blocking and self.rag_graph is not None
        if ready:
            summary = render_status(
                "success",
                "Ready for questions",
                "Ollama, required models, index, and graph are available.",
            )
        elif not blocking:
            summary = render_status(
                "info",
                "Application ready; AI not loaded",
                "Select Load AI models when you want to enable document questions.",
            )
        else:
            actions: list[str] = []
            checks = {row[0] for row in blocking}
            if "Ollama connectivity" in checks:
                actions.append(f"start Ollama with `ollama serve` at `{config.ollama_base_url}`")
            for label, model in (
                ("Chat model", config.llm_model),
                ("Embedding model", config.embedding_model),
            ):
                if label in checks:
                    actions.append(f"install `{model}` with `ollama pull {model}`")
            if checks - {"Ollama connectivity", "Chat model", "Embedding model"}:
                actions.append("review Diagnostics and rebuild or reconcile the local index")
            summary = render_status(
                "error", "Not ready for questions", "; ".join(actions) + "."
            )
        return (
            summary,
            self.diagnostic_display_rows(rows),
            gr.update(interactive=ready),
            gr.update(interactive=ready),
        )

    def load_ai_models(self, ollama: dict[str, Any] | None = None):
        """Initialize AI-backed services only after an explicit UI action."""
        rows = self.diagnostic_rows(ollama=ollama)
        if any(row[1] == "error" for row in rows):
            _, display_rows, message_update, send_update = self.preflight(ollama=ollama)
            return (
                render_status(
                    "error", "AI models not loaded", "Review Diagnostics, then retry."
                ),
                display_rows,
                message_update,
                send_update,
            )
        try:
            self.initialize()
        except Exception as exc:
            rows = self.diagnostic_rows(ollama=ollama)
            ai_row = next(index for index, row in enumerate(rows) if row[0] == "AI models")
            rows[ai_row] = ["AI models", "error", f"{type(exc).__name__}: {exc}"]
            return (
                render_status(
                    "error", "AI models not loaded", "Review Diagnostics, then retry."
                ),
                self.diagnostic_display_rows(rows),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        return self.preflight(ollama=ollama)

    @staticmethod
    def source_rows(result: dict[str, Any]) -> list[list[Any]]:
        hits = {hit.get("chunk_id"): hit for hit in result.get("retrieval_hits", [])}
        return [
            [
                source.get("label"),
                source.get("filename") or "—",
                source.get("page") if source.get("page") is not None else "—",
                source.get("excerpt") or "—",
                format_score(hits.get(source.get("chunk_id"), {}).get("semantic_score")),
                format_score(hits.get(source.get("chunk_id"), {}).get("sparse_score")),
                format_score(hits.get(source.get("chunk_id"), {}).get("fused_score")),
                format_score(hits.get(source.get("chunk_id"), {}).get("selection_score")),
            ]
            for source in result.get("sources", [])
        ]

    @staticmethod
    def trace_rows(result: dict[str, Any]) -> list[list[Any]]:
        return [
            [
                readable_label(event.get("stage")),
                readable_label(event.get("decision")),
                event.get("retrieved_count", event.get("candidate_count"))
                if event.get("retrieved_count", event.get("candidate_count")) is not None
                else "—",
                event.get("fused_count") if event.get("fused_count") is not None else "—",
                event.get("selected_count") if event.get("selected_count") is not None else "—",
                event.get("retry_count", 0),
                readable_label(event.get("termination")),
                format_duration_ms(event.get("duration_ms")),
            ]
            for event in result.get("trace", [])
        ]

    @staticmethod
    def score_rows(result: dict[str, Any]) -> list[list[Any]]:
        return [
            [
                hit.get("chunk_id"),
                hit.get("filename") or "—",
                hit.get("page") if hit.get("page") is not None else "—",
                format_score(hit.get("semantic_score")),
                format_score(hit.get("sparse_score")),
                format_score(hit.get("fused_score")),
                format_score(hit.get("selection_score")),
                ", ".join(hit.get("subqueries", [])) or "—",
            ]
            for hit in result.get("retrieval_hits", [])
        ]

    def chat(self, message: str, history: list[dict[str, str]], session_id: str):
        if not message.strip():
            return (
                "",
                history,
                {},
                render_status("info", "No answer yet", "Enter a question."),
                "Enter a question.",
                [],
                [],
                [],
            )
        try:
            result = self._graph().process_query(message.strip(), session_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            messages = history + [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": "The local RAG service is unavailable. Refresh Diagnostics and retry.",
                },
            ]
            return "", messages, {}, self.answer_status({}, error=error), error, [], [], []
        result.setdefault("standalone_query", message.strip())
        result["original_question"] = message.strip()
        diagnostics = (
            f"Original: **{message.strip()}** · Standalone: **{result['standalone_query']}** · "
            f"Route: **{result['route']}** · Strategy: **{result['strategy']}** · "
            f"Subqueries: **{', '.join(result.get('subqueries', [])) or 'none'}** · "
            f"Retrieval rounds: **{sum(e.get('stage') == 'retrieve' for e in result.get('trace', []))}** · "
            f"Retries: **{result['retry_count']}** · Evidence: **{result['evidence_status']}** · "
            f"Conflict: **{result.get('conflict', 'not reported')}**"
        )
        messages = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]},
        ]
        return (
            "",
            messages,
            result,
            self.answer_status(result),
            diagnostics,
            self.source_rows(result),
            self.score_rows(result),
            self.trace_rows(result),
        )

    def clear(self, session_id: str):
        if self.rag_graph is not None:
            self.rag_graph.clear(session_id)
        return (
            [],
            {},
            render_status("info", "No answer yet", "Conversation cleared."),
            "Conversation cleared.",
            [],
            [],
            [],
        )

    @staticmethod
    def answer_status(result: dict[str, Any], error: str | None = None) -> str:
        if error:
            return render_status("error", "Unavailable", f"The query could not complete: {error}")
        evidence = result.get("evidence_status")
        termination = next(
            (
                event.get("termination")
                for event in reversed(result.get("trace", []))
                if event.get("termination")
            ),
            None,
        )
        if evidence == "sufficient" or termination == "supported":
            return render_status(
                "success", "Supported", "The answer is backed by sufficient cited evidence."
            )
        if evidence == "limited" or termination == "limited":
            return render_status(
                "warning",
                "Limited",
                "Only part of the requested answer is supported by the evidence.",
            )
        if evidence == "insufficient" or termination in {"unsupported", "out_of_scope"}:
            return render_status(
                "warning",
                "Abstention",
                "The indexed evidence is insufficient for a grounded answer.",
            )
        return render_status(
            "info", "Completed", "Review the evidence and citations below."
        )

    @staticmethod
    def public_export(messages: list[Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": messages,
            "standalone_query": result.get("standalone_query"),
            "route": result.get("route"),
            "strategy": result.get("strategy"),
            "subqueries": result.get("subqueries", []),
            "retry_count": result.get("retry_count", 0),
            "evidence_status": result.get("evidence_status"),
            "citations": result.get("sources", []),
            "public_trace": result.get("trace", []),
        }

    def export_chat(self, messages: list[Any], result: dict[str, Any]) -> str:
        target = (
            config.data_dir
            / "exports"
            / f"conversation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.public_export(messages, result), indent=2) + "\n", encoding="utf-8"
        )
        return str(target)

    @staticmethod
    def load_evaluation_result(path: Path):
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        system_metrics = summary.get("metrics", {})
        metrics = [
            [
                label,
                *[
                    format_metric(name, system_metrics.get(system, {}).get(name))
                    for system in EVALUATION_SYSTEMS
                ],
            ]
            for name, label in DISPLAY_METRIC_LABELS.items()
        ]
        failures = []
        for line in (path / "cases.jsonl").read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            if case.get("failure_labels"):
                failures.append(
                    [
                        case.get("case_id") or "—",
                        readable_label(case.get("system")),
                        readable_label(case.get("route")),
                        readable_label(case.get("strategy")),
                        ", ".join(readable_label(label) for label in case["failure_labels"]),
                    ]
                )
        run_id = summary.get("configuration", {}).get("run_id", path.name)
        return metrics, failures, render_status(
            "success", "Evaluation loaded", f"{run_id} · {path}"
        )

    @staticmethod
    def latest_evaluation() -> Path | None:
        root = PROJECT_ROOT / "evals" / "results"
        candidates: list[tuple[float, Path]] = []
        for summary_path in root.rglob("summary.json"):
            result_path = summary_path.parent
            try:
                json.loads(summary_path.read_text(encoding="utf-8"))
                if not (result_path / "cases.jsonl").is_file():
                    continue
                candidates.append((summary_path.stat().st_mtime, result_path))
            except (json.JSONDecodeError, OSError):
                continue
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def load_latest_evaluation(self):
        latest = self.latest_evaluation()
        return (
            self.load_evaluation_result(latest)
            if latest
            else (
                [],
                [],
                render_status(
                    "info", "No saved evaluation", "Run an evaluation to create a result."
                ),
            )
        )

    def run_evaluation_ui(self, split: str, systems: list[str] | str | None):
        if not systems:
            return [], [], render_status(
                "warning", "No systems selected", "Select at least one evaluation system."
            )
        requested = [systems] if isinstance(systems, str) else list(systems)
        if not requested:
            return [], [], render_status(
                "warning", "No systems selected", "Select at least one evaluation system."
            )
        dataset = PROJECT_ROOT / "evals" / "mvp_cases.jsonl"
        if "REVIEW_REQUIRED_" in dataset.read_text(encoding="utf-8"):
            return (
                [],
                [],
                render_status(
                    "warning",
                    "Evaluation data needs review",
                    "Replace REVIEW_REQUIRED_* gold chunk IDs after indexing the reviewed corpus.",
                ),
            )
        selected = list(SYSTEMS) if "all" in requested else requested
        try:
            output = run_evaluation(dataset, selected, split)  # type: ignore[arg-type]
        except (RuntimeError, ValueError) as exc:
            return [], [], render_status("error", "Evaluation could not run", str(exc))
        return self.load_evaluation_result(output)

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        normalized = name.strip()
        return normalized if ":" in normalized else f"{normalized}:latest"

    def diagnostic_rows(self, ollama: dict[str, Any] | None = None) -> list[list[str]]:
        info = ollama or self._ollama_info()
        model_names = {self._normalize_model_name(name) for name in info["models"]}
        rows = [
            [
                "Ollama connectivity",
                "ok" if info["reachable"] else "error",
                "reachable" if info["reachable"] else f"Start Ollama at {config.ollama_base_url}",
            ]
        ]
        for label, model in (
            ("Chat model", config.llm_model),
            ("Embedding model", config.embedding_model),
        ):
            available = self._normalize_model_name(model) in model_names
            rows.append(
                [
                    label,
                    "ok" if available else "error",
                    model if available else f"Missing {model}; run: ollama pull {model}",
                ]
            )
        try:
            manifest = self.vector_db.manifest()
            reconciliation = self.vector_db.reconcile_index()
            rows.extend(
                [
                    ["Chroma collection", "ok", f"{self.vector_db.chunk_count()} chunks"],
                    ["Manifest", "ok", f"valid; {len(manifest.documents)} documents"],
                    [
                        "Missing Chroma chunks",
                        "ok" if not reconciliation.missing_chunk_ids else "error",
                        str(len(reconciliation.missing_chunk_ids)),
                    ],
                    [
                        "Orphan Chroma chunks",
                        "ok" if not reconciliation.orphan_chunk_ids else "warning",
                        str(len(reconciliation.orphan_chunk_ids)),
                    ],
                    [
                        "Duplicate IDs",
                        "ok" if not reconciliation.duplicate_chunk_ids else "error",
                        str(len(reconciliation.duplicate_chunk_ids)),
                    ],
                    [
                        "Missing source files",
                        "ok" if not reconciliation.missing_source_files else "warning",
                        str(len(reconciliation.missing_source_files)),
                    ],
                    [
                        "Index configuration",
                        "ok" if not reconciliation.incompatible_document_ids else "error",
                        "compatible"
                        if not reconciliation.incompatible_document_ids
                        else f"{len(reconciliation.incompatible_document_ids)} incompatible documents",
                    ],
                ]
            )
        except Exception as exc:
            rows.append(["Index diagnostics", "error", f"{type(exc).__name__}: {exc}"])
        rows.append(
            [
                "AI models",
                "ok" if self.rag_graph is not None else "pending",
                "loaded for this application session"
                if self.rag_graph is not None
                else "not loaded; use Load AI models when needed",
            ]
        )
        latest = self.latest_evaluation()
        rows.append(
            [
                "Latest evaluation",
                "ok" if latest else "pending",
                str(latest) if latest else "No stored result",
            ]
        )
        return rows

    @staticmethod
    def diagnostic_display_rows(rows: list[list[str]]) -> list[list[str]]:
        status_labels = {
            "ok": "Ready",
            "warning": "Review",
            "error": "Unavailable",
            "pending": "Not loaded",
        }
        return [
            [
                row[0],
                status_labels.get(row[1], readable_label(row[1])),
                row[2][:1].upper() + row[2][1:] if row[2] else "—",
            ]
            for row in rows
        ]

    def create_interface(self) -> gr.Blocks:
        theme = Soft(primary_hue="indigo", neutral_hue="slate")
        with gr.Blocks(
            title="Local Document RAG",
            theme=theme,
            css_paths=APP_STYLESHEET,
            js=ACCESSIBILITY_BOOTSTRAP,
            fill_width=True,
        ) as interface:
            session_id, latest_result = gr.State(lambda: str(uuid4())), gr.State({})
            gr.HTML(
                """
                <header class="app-header">
                  <div>
                    <p class="app-eyebrow">LOCAL · PRIVATE · GROUNDED</p>
                    <h1>Local Document RAG</h1>
                    <p>Manage your corpus, ask cited questions, and inspect the supporting evidence.</p>
                  </div>
                </header>
                """
            )
            readiness = gr.HTML(
                render_status(
                    "info",
                    "Interface ready without AI",
                    "Open Diagnostics when you want to enable document questions.",
                ),
                elem_id="readiness-status",
                elem_classes=["readiness-summary", "status-host"],
            )
            with gr.Tabs(elem_id="primary-tabs"):
                with gr.Tab("Documents"):
                    gr.Markdown(
                        "## Document library\nAdd local PDF or text files, then review what is available to search."
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=280):
                            files = gr.File(
                                label="PDF/TXT uploads",
                                file_count="multiple",
                                file_types=[".pdf", ".txt"],
                                type="filepath",
                                height=132,
                                elem_classes="upload-compact",
                            )
                            index_button = gr.Button(
                                "Index document", variant="primary", elem_classes="primary-action"
                            )
                            ingestion_status = gr.HTML(
                                render_status(
                                    "info", "Ready to index", "Select one or more files to begin."
                                ),
                                elem_id="ingestion-status",
                                elem_classes=["inline-status", "status-host"],
                            )
                        with gr.Column(scale=1, min_width=260):
                            gr.Markdown(
                                "### Library actions\nRefresh the table or review a document before removing it."
                            )
                            refresh_button = gr.Button("Refresh document status")
                            selected_id = gr.Dropdown(
                                label="Document ID to delete",
                                choices=[row[2] for row in self.document_rows()],
                            )
                            delete_button = gr.Button(
                                "Review deletion", elem_classes="destructive-review"
                            )
                    with gr.Group(
                        visible=False,
                        elem_id="deletion-alert",
                        elem_classes="delete-confirmation",
                    ) as delete_confirmation:
                        delete_confirmation_text = gr.HTML()
                        gr.Markdown(
                            "This removes the local document and its indexed chunks. This action cannot be undone."
                        )
                        with gr.Row(elem_classes="action-row"):
                            cancel_delete = gr.Button("Cancel")
                            confirm_delete = gr.Button("Confirm deletion", variant="stop")
                    documents = gr.Dataframe(
                        headers=DOCUMENT_HEADERS,
                        value=self.document_rows(),
                        label="Indexed documents",
                        interactive=False,
                        wrap=True,
                        show_search="filter",
                        max_height=360,
                        column_widths=[180, 220, 200, 80, 80, 130, 100, 240],
                        elem_id="documents-table",
                        elem_classes=["rag-table", "documents-table"],
                    )
                    with gr.Accordion("Maintenance", open=False):
                        gr.Markdown(
                            "Use these actions when source files change or the manifest and index need repair."
                        )
                        with gr.Row(elem_classes="action-row"):
                            reindex_button = gr.Button("Reindex changed documents")
                            reconcile_button = gr.Button("Reconcile manifest/index")
                            rebuild_button = gr.Button("Rebuild complete index")
                    with gr.Accordion("Indexing errors", open=False):
                        errors = gr.Dataframe(
                            headers=["Document", "Operation", "Error type", "Message"],
                            label="Indexing errors",
                            interactive=False,
                            wrap=True,
                            max_height=280,
                            elem_id="indexing-errors-table",
                            elem_classes=["rag-table", "indexing-errors-table"],
                        )
                with gr.Tab("Chat"):
                    gr.Markdown(
                        "## Ask your documents\nAnswers are grounded in the indexed corpus and linked to their cited sources."
                    )
                    answer_state = gr.HTML(
                        render_status("info", "No answer yet", "Ask a question to begin."),
                        elem_id="answer-status",
                        elem_classes=["answer-state", "status-host"],
                    )
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        type="messages",
                        allow_tags=False,
                        height=430,
                        elem_classes="conversation",
                    )
                    message = gr.Textbox(
                        label="Question",
                        placeholder="Load AI models in Diagnostics before asking about your documents",
                        interactive=False,
                        lines=2,
                    )
                    with gr.Row(elem_classes="action-row"):
                        send = gr.Button(
                            "Ask documents", variant="primary", interactive=False
                        )
                        clear = gr.Button("Clear conversation")
                        export = gr.Button("Export conversation JSON")
                    export_file = gr.File(label="Conversation export", height=72)
                    gr.Markdown("### Cited sources")
                    sources = gr.Dataframe(
                        headers=[
                            "Citation",
                            "Filename",
                            "Page",
                            "Excerpt",
                            "Semantic",
                            "Sparse",
                            "Fused",
                            "Selection",
                        ],
                        label="Sources used in this answer",
                        interactive=False,
                        wrap=True,
                        max_height=360,
                        column_widths=[90, 180, 70, 380, 100, 100, 100, 100],
                        elem_id="sources-table",
                        elem_classes=["rag-table", "sources-table"],
                    )
                    with gr.Accordion("Technical details", open=False):
                        evidence = gr.Markdown(
                            "Routing and evidence details will appear after a question."
                        )
                        scores = gr.Dataframe(
                            headers=SCORE_HEADERS,
                            label="Retrieval scores",
                            interactive=False,
                            wrap=True,
                            max_height=320,
                            elem_id="retrieval-scores-table",
                            elem_classes=["rag-table", "retrieval-scores-table"],
                        )
                        trace = gr.Dataframe(
                            headers=TRACE_HEADERS,
                            label="Retrieval trace",
                            interactive=False,
                            wrap=True,
                            max_height=320,
                            elem_id="retrieval-trace-table",
                            elem_classes=["rag-table", "retrieval-trace-table"],
                        )
                with gr.Tab("Evaluation"):
                    gr.Markdown(
                        "## Evaluation results\nRun the existing benchmark or inspect the newest saved result."
                    )
                    with gr.Row(elem_classes="action-row"):
                        split = gr.Dropdown(
                            ["development", "test"], value="development", label="Split"
                        )
                        systems = gr.CheckboxGroup(
                            [*SYSTEMS, "all"],
                            value=["dense", "bm25", "hybrid", "agentic"],
                            label="Systems",
                        )
                    with gr.Row():
                        run_eval = gr.Button("Run evaluation", variant="primary")
                        load_eval = gr.Button("Load latest result")
                    eval_status = gr.HTML(
                        render_status(
                            "info", "Evaluation ready", "Choose systems and a split."
                        ),
                        elem_id="evaluation-status",
                        elem_classes=["inline-status", "status-host"],
                    )
                    metrics = gr.Dataframe(
                        headers=EVALUATION_HEADERS,
                        label="Metrics comparison",
                        interactive=False,
                        wrap=True,
                        max_height=360,
                        column_widths=[260, 120, 120, 120, 120],
                        elem_id="evaluation-metrics-table",
                        elem_classes=["rag-table", "evaluation-metrics-table"],
                    )
                    failures = gr.Dataframe(
                        headers=["Case", "System", "Route", "Strategy", "Failure labels"],
                        label="Failure cases",
                        interactive=False,
                        wrap=True,
                        show_search="filter",
                        max_height=360,
                        column_widths=[170, 110, 130, 130, 360],
                        elem_id="evaluation-failures-table",
                        elem_classes=["rag-table", "evaluation-failures-table"],
                    )
                with gr.Tab("Diagnostics"):
                    gr.Markdown(
                        "## Local readiness\nThe interface runs independently from Ollama. "
                        "Inspect availability, then load AI models only when needed."
                    )
                    with gr.Row(elem_classes="action-row"):
                        load_ai = gr.Button("Load AI models", variant="primary")
                        refresh_diagnostics = gr.Button("Refresh status")
                    diagnostics = gr.Dataframe(
                        headers=["Check", "Status", "Details"],
                        label="Readiness checks",
                        interactive=False,
                        wrap=True,
                        max_height=480,
                        column_widths=[200, 100, 520],
                        elem_id="diagnostics-table",
                        elem_classes=["rag-table", "diagnostics-table"],
                    )

            index_button.click(
                self.index_selected,
                files,
                [documents, ingestion_status, errors, readiness, selected_id],
            )
            document_outputs = [documents, ingestion_status, errors, selected_id]
            reindex_button.click(self.reindex_changed, None, document_outputs)
            rebuild_button.click(self.rebuild_index, None, document_outputs)
            reconcile_button.click(self.reconcile_manifest_index, None, document_outputs)
            delete_button.click(
                self.prepare_deletion,
                selected_id,
                [delete_confirmation_text, delete_confirmation],
            )
            cancel_delete.click(
                self.cancel_deletion,
                None,
                [delete_confirmation_text, delete_confirmation],
            )
            selected_id.change(
                self.cancel_deletion,
                None,
                [delete_confirmation_text, delete_confirmation],
            )
            confirm_delete.click(
                self.delete_selected,
                selected_id,
                [*document_outputs, delete_confirmation_text, delete_confirmation],
            )
            refresh_button.click(self.refresh_documents, None, [documents, readiness, selected_id])
            for trigger in (send.click, message.submit):
                trigger(
                    self.chat,
                    [message, chatbot, session_id],
                    [
                        message,
                        chatbot,
                        latest_result,
                        answer_state,
                        evidence,
                        sources,
                        scores,
                        trace,
                    ],
                )
            clear.click(
                self.clear,
                session_id,
                [chatbot, latest_result, answer_state, evidence, sources, scores, trace],
            )
            export.click(self.export_chat, [chatbot, latest_result], export_file)
            run_eval.click(
                self.run_evaluation_ui, [split, systems], [metrics, failures, eval_status]
            )
            load_eval.click(self.load_latest_evaluation, None, [metrics, failures, eval_status])
            preflight_outputs = [readiness, diagnostics, message, send]
            refresh_diagnostics.click(self.preflight, None, preflight_outputs)
            load_ai.click(self.load_ai_models, None, preflight_outputs)
        return interface


app = RAGApplication()


def main() -> int:
    app.create_interface().launch(
        server_name=config.gradio_host, server_port=config.gradio_port, share=config.gradio_share
    )
    return 0


if __name__ == "__main__":
    main()
