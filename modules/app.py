"""Gradio interface exposing the existing local RAG services."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from html import escape
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import gradio as gr
from gradio.themes import Soft

from modules.config import PROJECT_ROOT, config
from modules.evaluation import (
    SYSTEMS,
    load_cases,
    normalize_model_name,
    preflight_multihop,
    required_models_for_systems,
    run_evaluation,
)
from modules.rag_graph import RAGGraph
from modules.vector_db import VectorDBManager

DOCUMENT_HEADERS = ["Document", "Pages", "Chunks", "Status"]
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
    "document_recall_at_5",
    "route_accuracy",
    "strategy_accuracy",
    "retry_precision",
    "retry_recall",
    "citation_precision",
    "gold_evidence_citation_coverage",
    "abstention_accuracy",
    "unanswerable_abstention_recall",
    "answerable_response_rate",
    "conflict_recall",
    "conflict_false_positive_rate",
    "normalized_answer_exact_match",
    "answer_token_f1",
    "termination_rate",
    "mean_latency_seconds",
    "p95_latency_seconds",
    "mean_llm_calls_per_query",
    "mean_retrieval_rounds_per_query",
]
DISPLAY_METRIC_LABELS = {
    "recall_at_5": "Recall at 5",
    "mrr_at_5": "MRR at 5",
    "ndcg_at_5": "NDCG at 5",
    "document_recall_at_5": "Document recall at 5",
    "citation_precision": "Citation precision",
    "gold_evidence_citation_coverage": "Gold evidence citation coverage",
    "abstention_accuracy": "Abstention accuracy",
    "unanswerable_abstention_recall": "Unanswerable abstention recall",
    "answerable_response_rate": "Answerable response rate",
    "conflict_recall": "Conflict recall",
    "conflict_false_positive_rate": "Conflict false positive rate",
    "normalized_answer_exact_match": "Normalized answer exact match",
    "answer_token_f1": "Answer token F1",
    "p95_latency_seconds": "P95 latency",
    "mean_llm_calls_per_query": "Mean LLM calls per query",
    "mean_retrieval_rounds_per_query": "Mean retrieval rounds per query",
}
METRIC_GROUPS = (
    (
        "Retrieval",
        ("recall_at_5", "mrr_at_5", "ndcg_at_5", "document_recall_at_5"),
    ),
    (
        "Evidence and grounding",
        (
            "citation_precision",
            "gold_evidence_citation_coverage",
            "abstention_accuracy",
            "unanswerable_abstention_recall",
            "answerable_response_rate",
            "conflict_recall",
            "conflict_false_positive_rate",
        ),
    ),
    ("Answer quality", ("answer_token_f1", "normalized_answer_exact_match")),
    (
        "Workflow cost",
        (
            "p95_latency_seconds",
            "mean_llm_calls_per_query",
            "mean_retrieval_rounds_per_query",
        ),
    ),
)
HIDDEN_METRICS = {
    "mean_latency_seconds",
    "route_accuracy",
    "strategy_accuracy",
    "retry_precision",
    "retry_recall",
    "termination_rate",
    "chunk_recall_at_5",
    "conflict_accuracy",
}

EvaluationState = Literal["ready", "blocked", "running", "result", "error"]


@dataclass(frozen=True)
class EvaluationReadiness:
    state: EvaluationState
    latest_result: Path | None = None
    systems: tuple[str, ...] = ()
    split: str = "development"
    requires_index: bool = True
    requires_embeddings: bool = False
    requires_chat: bool = False
    problems: tuple[str, ...] = ()
DISPLAY_METRIC_NAMES = [name for _, names in METRIC_GROUPS for name in names]
EVALUATION_SYSTEMS = ("dense", "bm25", "hybrid", "agentic")
EVALUATION_HEADERS = ["Category", "Metric", "Dense", "BM25", "Hybrid", "Agentic"]
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
    "unanswerable_abstention_recall",
    "answerable_response_rate",
    "conflict_recall",
    "conflict_false_positive_rate",
    "normalized_answer_exact_match",
    "answer_token_f1",
    "termination_rate",
}
AGENTIC_ONLY_METRICS = {
    "route_accuracy",
    "strategy_accuracy",
    "retry_precision",
    "retry_recall",
    "citation_precision",
    "gold_evidence_citation_coverage",
    "abstention_accuracy",
    "unanswerable_abstention_recall",
    "answerable_response_rate",
    "conflict_recall",
    "conflict_false_positive_rate",
    "normalized_answer_exact_match",
    "answer_token_f1",
}
_MISSING_METRIC = object()
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
    if name in PERCENTAGE_METRICS or (
        name not in METRIC_NAMES and 0 <= float(value) <= 1
    ):
        return f"{float(value) * 100:.1f}%"
    if name == "p95_latency_seconds":
        return format_duration_ms(float(value) * 1000)
    rounded = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{rounded:.1f}"


def format_metric_observation(
    name: str,
    raw: Any,
    *,
    system: str,
    schema_version: int,
    system_present: bool = True,
) -> str:
    """Format a schema-v2 metric observation for the comparison matrix."""
    if schema_version != 2:
        raise ValueError(
            "Evaluation summaries must use schema version 2. Run a new evaluation."
        )
    if not system_present:
        return "—"
    if raw is _MISSING_METRIC:
        return "—"
    if not isinstance(raw, dict):
        return "—"

    status = raw.get("status")
    if status == "not_applicable":
        return "— Not applicable"
    if status == "no_eligible_cases":
        return "— No eligible cases"
    if status != "measured" or raw.get("value") is None:
        return "—"
    sample_count = raw.get("sample_count")
    support = f" · n={sample_count}" if isinstance(sample_count, int) else ""
    return f"{format_metric(name, raw['value'])}{support}"


def render_status(kind: StatusKind, title: str, detail: str = "") -> str:
    icons = {"info": "i", "success": "✓", "warning": "!", "error": "×"}
    safe_title = escape(title)
    safe_detail = escape(detail)
    detail_html = f'<span class="rag-status__detail">{safe_detail}</span>' if detail else ""
    role = "alert" if kind == "error" else "status"
    live = "assertive" if kind == "error" else "polite"
    return (
        f'<div class="rag-status rag-status--{kind}" role="{role}" '
        f'aria-live="{live}" aria-atomic="true">'
        f'<span class="rag-status__icon" aria-hidden="true">{icons[kind]}</span>'
        '<span class="rag-status__copy">'
        f'<strong class="rag-status__title">{safe_title}</strong>{detail_html}'
        "</span></div>"
    )


def render_result_table(
    headers: Sequence[Any],
    rows: Sequence[Sequence[Any]] | None,
    *,
    caption: str,
    empty_message: str,
    mobile_cards: bool = False,
    table_class: str = "",
) -> str:
    """Render read-only results without inheriting Gradio's editable grid UI."""
    safe_headers = [escape(str(header)) for header in headers]
    safe_caption = escape(caption)
    safe_empty = escape(empty_message)
    row_values = list(rows or [])
    section_classes = "result-view"
    if table_class:
        section_classes += f" {escape(table_class)}"
    if not row_values:
        return (
            f'<section class="{section_classes}" aria-label="{safe_caption}">'
            f'<div class="result-empty" role="status">{safe_empty}</div>'
            "</section>"
        )

    status_kinds = {
        "ready": "success",
        "indexed": "success",
        "review": "warning",
        "warning": "warning",
        "unavailable": "error",
        "error": "error",
        "failed": "error",
        "not loaded": "info",
        "pending": "info",
    }
    card_class = " stack-on-mobile" if mobile_cards else ""
    body = []
    for row in row_values:
        cells = []
        for index, header in enumerate(safe_headers):
            raw_value = row[index] if index < len(row) else "—"
            display_value = "—" if raw_value is None or raw_value == "" else str(raw_value)
            safe_value = escape(display_value)
            status = status_kinds.get(display_value.strip().lower())
            status_attribute = f' data-status="{status}"' if status else ""
            cells.append(
                f'<td data-label="{header}"{status_attribute}>{safe_value}</td>'
            )
        body.append(f"<tr>{''.join(cells)}</tr>")

    headings = "".join(f'<th scope="col">{header}</th>' for header in safe_headers)
    return (
        f'<section class="{section_classes}">'
        '<div class="result-scroll">'
        f'<table class="result-table{card_class}">'
        f"<caption>{safe_caption}</caption>"
        f"<thead><tr>{headings}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div></section>"
    )


def normalize_result_rows(value: Any) -> list[list[Any]]:
    """Narrow Gradio's broad component value type to safe tabular rows."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [
        list(row)
        for row in value
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
    ]


ACCESSIBILITY_BOOTSTRAP = """
() => {
  const regionLabels = {
    "documents-table": "Indexed documents",
    "indexing-errors-table": "Indexing errors",
    "conversation-region": "Conversation",
    "evidence-list": "Cited evidence",
    "retrieval-scores-table": "Retrieval scores",
    "retrieval-trace-table": "Retrieval trace",
    "evaluation-metrics-table": "Evaluation metrics comparison",
    "evaluation-failures-table": "Evaluation failure cases",
    "system-status-details": "System status details",
  };
  const clearScrollSemantics = (target) => {
    target.removeAttribute("role");
    target.removeAttribute("aria-label");
    target.removeAttribute("tabindex");
    delete target.dataset.overflowX;
    delete target.dataset.overflowY;
  };
  const syncRegion = (region) => {
    const target = region.querySelector(".result-scroll, .wrap") || region;
    const previousTarget = region._ragScrollTarget;
    if (previousTarget && previousTarget !== target) clearScrollSemantics(previousTarget);
    region._ragScrollTarget = target;
    const overflowX = target.scrollWidth > target.clientWidth + 1;
    const overflowY = target.scrollHeight > target.clientHeight + 1;
    const nextOverflowX = String(overflowX);
    const nextOverflowY = String(overflowY);
    if (target.dataset.overflowX !== nextOverflowX) {
      target.dataset.overflowX = nextOverflowX;
    }
    if (target.dataset.overflowY !== nextOverflowY) {
      target.dataset.overflowY = nextOverflowY;
    }
    if (overflowX || overflowY) {
      const directions = [
        overflowX && "horizontally scrollable",
        overflowY && "vertically scrollable",
      ]
        .filter(Boolean)
        .join(" and ");
      const label = `${regionLabels[region.id] || "Scrollable results"}, ${directions} scrolling available`;
      if (target.getAttribute("role") !== "region") {
        target.setAttribute("role", "region");
      }
      if (target.getAttribute("aria-label") !== label) {
        target.setAttribute("aria-label", label);
      }
      if (target.getAttribute("tabindex") !== "0") {
        target.setAttribute("tabindex", "0");
      }
    } else {
      target.removeAttribute("role");
      target.removeAttribute("aria-label");
      target.removeAttribute("tabindex");
    }
    return target;
  };
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const region = entry.target.closest(".overflow-region");
      if (region) syncRegion(region);
    }
  });
  const syncAccordion = (accordion) => {
    const trigger = accordion.querySelector("button.label-wrap");
    if (!trigger) return;
    const expanded = accordion.classList.contains("open");
    const nextExpanded = String(expanded);
    if (trigger.getAttribute("aria-expanded") !== nextExpanded) {
      trigger.setAttribute("aria-expanded", nextExpanded);
    }
  };
  const enhanceNode = (node) => {
    if (!(node instanceof Element)) return;
    const deletion = document.getElementById("deletion-alert");
    if (deletion) {
      deletion.setAttribute("role", "alert");
      deletion.setAttribute("aria-live", "assertive");
    }
    const regions = node.matches(".overflow-region")
      ? [node]
      : Array.from(node.querySelectorAll(".overflow-region"));
    for (const region of regions) {
      const target = syncRegion(region);
      if (!target.dataset.resizeObserved) {
        target.dataset.resizeObserved = "true";
        resizeObserver.observe(target);
      }
    }
    const accordions = node.matches(".gradio-accordion")
      ? [node]
      : Array.from(node.querySelectorAll(".gradio-accordion"));
    for (const accordion of accordions) syncAccordion(accordion);
  };
  enhanceNode(document.body);
  const pending = new Set();
  let frame = null;
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      const parentRegion = mutation.target instanceof Element
        ? mutation.target.closest(".overflow-region")
        : null;
      if (parentRegion) pending.add(parentRegion);
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

  requestAnimationFrame(() => enhanceNode(document.body));
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
                record.relative_path,
                record.page_count,
                record.chunk_count,
                "Review" if self.last_errors.get(record.document_id) else "Indexed",
            ]
            for record in sorted(manifest.documents.values(), key=lambda item: item.relative_path)
        ]

    def document_samples(self, query: str | None = None) -> list[list[Any]]:
        """Return read-only inventory samples without exposing stable document IDs."""
        needle = str(query or "").strip().casefold()
        rows = self.document_rows()
        if not needle:
            return rows
        return [
            row
            for row in rows
            if needle in str(row[0]).casefold() or needle in str(row[3]).casefold()
        ]

    def corpus_summary_html(self) -> str:
        rows = self.document_rows()
        pages = sum(int(row[1] or 0) for row in rows)
        chunks = sum(int(row[2] or 0) for row in rows)
        return (
            '<div class="corpus-summary" role="status" aria-live="polite">'
            f'<strong>{len(rows)} document(s)</strong>'
            f'<span>{pages} page(s) · {chunks} chunk(s)</span>'
            "</div>"
        )

    def current_error_rows(self) -> list[list[str]]:
        manifest = self.vector_db.manifest()
        rows: list[list[str]] = []
        for document_id, message in sorted(self.last_errors.items()):
            record = manifest.documents.get(document_id)
            rows.append(
                [
                    record.relative_path if record else "Unknown document",
                    "Index",
                    "Indexing error",
                    message or "—",
                ]
            )
        return rows

    def select_document(self, rows_or_query: Any, event: gr.SelectData):
        normalized = (
            self.document_samples(rows_or_query)
            if rows_or_query is None or isinstance(rows_or_query, str)
            else normalize_result_rows(rows_or_query)
        )
        index = event.index[0] if isinstance(event.index, (tuple, list)) else event.index
        if not isinstance(index, int) or index < 0 or index >= len(normalized):
            return self.reset_document_selection(normalized)
        relative_path = str(normalized[index][0])
        record = next(
            (
                item
                for item in self.vector_db.manifest().documents.values()
                if item.relative_path == relative_path
            ),
            None,
        )
        if record is None:
            return self.reset_document_selection(normalized)
        summary = (
            '<div class="selected-document">'
            f'<strong>{escape(record.relative_path)}</strong>'
            f'<span>{record.page_count} page(s) · {record.chunk_count} chunk(s)</span>'
            "</div>"
        )
        return (
            record.document_id,
            gr.update(value=summary, visible=True),
            gr.update(visible=True, interactive=True),
            "",
            gr.update(visible=False),
        )

    def reset_document_selection(self, rows: Any = None):
        normalized = self.document_rows() if rows is None else normalize_result_rows(rows)
        return (
            "",
            gr.update(value="", visible=False),
            gr.update(visible=bool(normalized), interactive=False),
            "",
            gr.update(visible=False),
        )

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

    @staticmethod
    def indexing_errors_html(rows: Sequence[Sequence[Any]] | None) -> str:
        return render_result_table(
            ["Document", "Operation", "Error type", "Message"],
            rows,
            caption="Indexing errors",
            empty_message="No indexing errors.",
            mobile_cards=True,
            table_class="indexing-errors-view",
        )

    @staticmethod
    def scores_html(rows: Sequence[Sequence[Any]] | None) -> str:
        return render_result_table(
            SCORE_HEADERS,
            rows,
            caption="Retrieval scores",
            empty_message="No retrieval scores yet.",
            mobile_cards=True,
            table_class="retrieval-scores-view",
        )

    @staticmethod
    def trace_html(rows: Sequence[Sequence[Any]] | None) -> str:
        return render_result_table(
            TRACE_HEADERS,
            rows,
            caption="Retrieval trace",
            empty_message="No retrieval trace yet.",
            mobile_cards=True,
            table_class="retrieval-trace-view",
        )

    @staticmethod
    def metrics_html(rows: Sequence[Sequence[Any]] | None) -> str:
        normalized = normalize_result_rows(rows)
        if not normalized:
            return (
                '<section class="result-view evaluation-metrics-view" '
                'aria-label="Metrics comparison">'
                '<div class="result-empty" role="status">'
                "No evaluation metrics loaded.</div></section>"
            )

        body = []
        for row in normalized:
            category = escape(str(row[0] if row else "—"))
            metric = escape(str(row[1] if len(row) > 1 else "—"))
            cells = [
                f'<td data-label="Category">{category}</td>',
                f'<th scope="row" data-label="Metric">{metric}</th>',
            ]
            for index, header in enumerate(EVALUATION_HEADERS[2:], start=2):
                raw_value = str(row[index] if index < len(row) else "—")
                primary, support = RAGApplication._metric_value_parts(raw_value)
                neutral = " metric-value--neutral" if primary == "—" else ""
                support_html = (
                    f'<span class="metric-support">{escape(support)}</span>'
                    if support
                    else ""
                )
                cells.append(
                    f'<td data-label="{escape(header)}">'
                    f'<span class="metric-value{neutral}">{escape(primary)}</span>'
                    f"{support_html}</td>"
                )
            body.append(f"<tr>{''.join(cells)}</tr>")

        headings = "".join(
            f'<th scope="col">{escape(header)}</th>' for header in EVALUATION_HEADERS
        )
        return (
            '<section class="result-view evaluation-metrics-view">'
            '<div class="result-scroll">'
            '<table class="result-table evaluation-matrix">'
            '<caption>Metrics comparison</caption>'
            f"<thead><tr>{headings}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody>"
            "</table></div></section>"
        )

    @staticmethod
    def _metric_value_parts(value: str) -> tuple[str, str]:
        if value.startswith("— "):
            detail = value[2:]
            return "—", detail
        if " · " in value:
            primary, support = value.split(" · ", 1)
            return primary, support
        return value, ""

    @staticmethod
    def failures_html(rows: Sequence[Sequence[Any]] | None) -> str:
        return render_result_table(
            ["Case", "System", "Route", "Strategy", "Failure labels"],
            rows,
            caption="Failure cases",
            empty_message="No evaluation failures to display.",
            mobile_cards=True,
            table_class="evaluation-failures-view",
        )

    @staticmethod
    def system_status_html(rows: Sequence[Sequence[Any]] | None) -> str:
        normalized = normalize_result_rows(rows)
        if not normalized:
            technical = render_result_table(
                ["Area", "Check", "Status", "Details"],
                [],
                caption="Technical system values",
                empty_message="Workspace checks have not run yet.",
                mobile_cards=True,
                table_class="system-status-view",
            )
            return (
                '<section class="system-status" aria-label="System status" '
                'role="status" aria-live="polite">'
                '<div class="system-status__summary system-status__summary--unknown">'
                '<strong>Status unknown</strong>'
                '<span>Workspace checks have not run yet.</span></div>'
                '<h4 class="system-status__technical-title">Technical values</h4>'
                f"{technical}</section>"
            )
        problem_rows = [
            row for row in normalized if len(row) >= 4 and str(row[2]) != "Ready"
        ]
        if problem_rows:
            has_error = any(
                str(row[2]).casefold() in {"error", "unavailable"}
                for row in problem_rows
            )
            groups: dict[str, list[list[Any]]] = {}
            for row in problem_rows:
                groups.setdefault(str(row[0]), []).append(row)
            state = "error" if has_error else "warning"
            title = "Action required" if has_error else "Review recommended"
            overview = [
                f'<div class="system-status__summary system-status__summary--{state}">'
                f"<strong>{title}</strong>"
            ]
            for category, items in groups.items():
                overview.append(f'<section><h4>{escape(category)}</h4><ul>')
                for item in items:
                    overview.append(
                        f'<li><strong>{escape(str(item[1]))}</strong>'
                        f'<span>{escape(str(item[3]))}</span></li>'
                    )
                overview.append("</ul></section>")
            overview.append("</div>")
            summary = "".join(overview)
        else:
            summary = (
                '<div class="system-status__summary system-status__summary--ready">'
                '<strong>System ready</strong>'
                '<span>Local services, models, and indexed data are available.</span></div>'
            )
        technical = render_result_table(
            ["Area", "Check", "Status", "Details"],
            normalized,
            caption="Technical system values",
            empty_message="Workspace checks have not run yet.",
            mobile_cards=True,
            table_class="system-status-view",
        )
        role = "alert" if any(
            str(row[2]).casefold() in {"error", "unavailable"} for row in problem_rows
        ) else "status"
        return (
            '<section class="system-status" aria-label="System status" '
            f'role="{role}" aria-live="polite">{summary}'
            '<h4 class="system-status__technical-title">Technical values</h4>'
            f"{technical}</section>"
        )

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
            elif result.success:
                self.last_errors.pop(result.document_id, None)
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
        )

    def delete_selected(self, document_id: str | None):
        if not document_id:
            return (
                self.document_rows(),
                render_status("warning", "No document selected", "Select a document to delete."),
                [],
                "",
                gr.update(visible=False),
            )
        record = self.vector_db.manifest().documents.get(document_id)
        display_name = record.relative_path if record else "Selected document"
        deleted = self.vector_db.delete_document(document_id)
        self._reset_graph()
        if deleted:
            self.last_errors.pop(document_id, None)
        status = render_status(
            "success" if deleted else "warning",
            "Deleted document" if deleted else "Document not found",
            display_name,
        )
        return (
            self.document_rows(),
            status,
            [],
            "",
            gr.update(visible=False),
        )

    def prepare_deletion(self, document_id: str | None):
        if not document_id:
            return "", gr.update(visible=False)
        record = self.vector_db.manifest().documents.get(document_id)
        filename = record.relative_path if record else "selected document"
        return (
            f"<strong>Delete {escape(filename)}</strong> and its indexed chunks?",
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
        self.last_errors.clear()
        return (
            self.document_rows(),
            render_status("success", "Rebuilt complete index", f"{count} chunks available"),
            [],
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
        return self.document_rows(), status, []

    def refresh_documents(self):
        return self.document_rows(), self.readiness()

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
                actions.append("review System status and rebuild the local index if needed")
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
                    "error", "AI models not loaded", "Open System status, then retry."
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
                    "error", "AI models not loaded", "Open System status, then retry."
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
    def evidence_html(result: dict[str, Any]) -> str:
        sources = result.get("sources", [])
        if not sources:
            return (
                '<section class="evidence-list evidence-list--empty" '
                'aria-label="Cited evidence">'
                '<p class="empty-result">No cited evidence yet. Sources will appear after a supported answer.</p>'
                "</section>"
            )
        items = []
        for source in sources:
            label = escape(str(source.get("label") or "Source"))
            filename = escape(str(source.get("filename") or "Unknown file"))
            page = source.get("page")
            location = f"{filename} · Page {escape(str(page))}" if page is not None else filename
            excerpt = escape(str(source.get("excerpt") or "No excerpt available."))
            relevant = source.get("relevant", True)
            relevant_badge = (
                '<span class="evidence-state evidence-state--relevant">Relevant</span>'
                if relevant
                else ""
            )
            items.append(
                '<article class="evidence-item">'
                '<div class="evidence-item__header">'
                f'<span class="evidence-citation">{label}</span>'
                f'<strong class="evidence-location">{location}</strong>'
                '<span class="evidence-state evidence-state--cited">Cited</span>'
                f"{relevant_badge}</div>"
                f'<blockquote class="evidence-excerpt">{excerpt}</blockquote>'
                "</article>"
            )
        return (
            '<section class="evidence-list" aria-label="Cited evidence">'
            + "".join(items)
            + "</section>"
        )

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
                self.evidence_html({}),
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
                    "content": "The local RAG service is unavailable. Review System status and retry.",
                },
            ]
            return (
                "",
                messages,
                {},
                self.answer_status({}, error=error),
                error,
                self.evidence_html({}),
                [],
                [],
            )
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
            self.evidence_html(result),
            self.score_rows(result),
            self.trace_rows(result),
        )

    def clear(self, session_id: str):
        if self.rag_graph is not None:
            self.rag_graph.clear(session_id)
        return (
            self.evidence_html({}),
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
            "validation": result.get("validation"),
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
    def evaluation_context_html(
        path: Path,
        summary: dict[str, Any],
        case_count: int,
    ) -> str:
        configuration = summary.get("configuration", {})
        if not isinstance(configuration, dict):
            configuration = {}
        run_id = configuration.get("run_id") or path.name
        dataset = configuration.get("dataset_name") or path.parent.name or "—"
        split = configuration.get("evaluated_split") or "—"
        configured_systems = configuration.get("systems")
        if isinstance(configured_systems, list):
            systems = [str(system) for system in configured_systems]
        else:
            metrics = summary.get("metrics", {})
            systems = list(metrics) if isinstance(metrics, dict) else []
        system_labels = {
            "dense": "Dense",
            "bm25": "BM25",
            "hybrid": "Hybrid",
            "agentic": "Agentic",
        }
        systems_text = ", ".join(
            system_labels.get(system, readable_label(system)) for system in systems
        ) or "—"
        raw_timestamp = configuration.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
            timestamp = timestamp.astimezone(UTC)
        except (TypeError, ValueError):
            timestamp = datetime.fromtimestamp((path / "summary.json").stat().st_mtime, UTC)
        case_label = f"{case_count} case" if case_count == 1 else f"{case_count} cases"
        items = (
            ("Run", run_id),
            ("Dataset", f"{dataset} · {split}"),
            ("Systems", systems_text),
            ("Cases", case_label),
            ("Result date", timestamp.strftime("%Y-%m-%d %H:%M UTC")),
        )
        content = "".join(
            '<div class="evaluation-context__item">'
            f'<span>{escape(str(label))}</span><strong>{escape(str(value))}</strong>'
            "</div>"
            for label, value in items
        )
        return (
            '<section class="evaluation-context" aria-label="Evaluation result context">'
            f"{content}</section>"
        )

    @staticmethod
    def load_evaluation_result(path: Path):
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("schema_version") != 2:
            raise ValueError(
                "This saved evaluation predates schema version 2. "
                "Run a new evaluation to create a compatible result."
            )
        system_metrics = summary.get("metrics", {})
        schema_version = 2
        metrics = []
        for category, names in METRIC_GROUPS:
            for name in names:
                metrics.append(
                    [
                        category,
                        DISPLAY_METRIC_LABELS[name],
                        *[
                            format_metric_observation(
                                name,
                                system_metrics.get(system, {}).get(name, _MISSING_METRIC),
                                system=system,
                                schema_version=schema_version,
                                system_present=system in system_metrics,
                            )
                            for system in EVALUATION_SYSTEMS
                        ],
                    ]
                )
        available_names = {
            name for values in system_metrics.values() for name in values
        }
        for name in sorted(available_names - set(DISPLAY_METRIC_NAMES) - HIDDEN_METRICS):
            metrics.append(
                [
                    "Other",
                    readable_label(name),
                    *[
                        format_metric_observation(
                            name,
                            system_metrics.get(system, {}).get(name, _MISSING_METRIC),
                            system=system,
                            schema_version=schema_version,
                            system_present=system in system_metrics,
                        )
                        for system in EVALUATION_SYSTEMS
                    ],
                ]
            )
        failures = []
        case_lines = [
            line
            for line in (path / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for line in case_lines:
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
        context = RAGApplication.evaluation_context_html(path, summary, len(case_lines))
        return metrics, failures, context, render_status(
            "success", "Evaluation loaded", f"{run_id} · {path}"
        )

    @staticmethod
    def latest_evaluation() -> Path | None:
        root = PROJECT_ROOT / "evals" / "results" / "multihop"
        candidates: list[tuple[float, Path]] = []
        for summary_path in root.rglob("summary.json"):
            result_path = summary_path.parent
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("schema_version") != 2:
                    continue
                if summary.get("configuration", {}).get("dataset_name") != "multihop":
                    continue
                if not (result_path / "cases.jsonl").is_file():
                    continue
                for line in (result_path / "cases.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines():
                    if line.strip():
                        json.loads(line)
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
                "",
                render_status(
                    "info",
                    "No compatible saved evaluation",
                    "Run a new evaluation to create a schema version 2 result.",
                ),
            )
        )

    def evaluation_readiness(
        self,
        split: str,
        systems: Sequence[str] | str | None,
        *,
        ollama: dict[str, Any] | None = None,
    ) -> EvaluationReadiness:
        requested = [systems] if isinstance(systems, str) else list(systems or [])
        selected = tuple(system for system in requested if system in SYSTEMS)
        problems: list[str] = []
        dataset = PROJECT_ROOT / "evals" / "multihop" / "cases.jsonl"
        if not selected:
            problems.append("Select at least one system in Advanced options.")
        if split not in {"development", "test"}:
            problems.append("Choose the development or held-out test split.")
        if not dataset.is_file():
            problems.append(
                "Prepare MultiHopRAG with: uv run python scripts/prepare_multihop_eval.py --index"
            )
        elif selected:
            try:
                preflight_multihop(load_cases(dataset), check_models=False)
            except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
                problems.append(str(exc))
        required = required_models_for_systems(selected)  # type: ignore[arg-type]
        if required:
            info = ollama or self._ollama_info()
            if not info.get("reachable"):
                problems.append("Start Ollama with: ollama serve")
            else:
                available = {
                    normalize_model_name(name) for name in info.get("models", [])
                }
                missing = [name for name in required if name not in available]
                if missing:
                    problems.append(
                        "Install the required model with: "
                        + " && ".join(f"ollama pull {name}" for name in missing)
                    )
        latest = self.latest_evaluation()
        return EvaluationReadiness(
            state="blocked" if problems else ("result" if latest else "ready"),
            latest_result=latest,
            systems=selected,
            split=split,
            requires_embeddings=any(
                system in {"dense", "hybrid", "agentic"} for system in selected
            ),
            requires_chat="agentic" in selected,
            problems=tuple(dict.fromkeys(problems)),
        )

    def run_evaluation_ui(self, split: str, systems: list[str] | str | None):
        if not systems:
            return [], [], "", render_status(
                "warning", "No systems selected", "Select at least one evaluation system."
            )
        requested = [systems] if isinstance(systems, str) else list(systems)
        if not requested:
            return [], [], "", render_status(
                "warning", "No systems selected", "Select at least one evaluation system."
            )
        dataset = PROJECT_ROOT / "evals" / "multihop" / "cases.jsonl"
        selected = requested
        try:
            output = run_evaluation(
                dataset, selected, split, dataset_name="multihop"  # type: ignore[arg-type]
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return [], [], "", render_status("error", "Evaluation could not run", str(exc))
        return self.load_evaluation_result(output)

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return normalize_model_name(name)

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

    @staticmethod
    def diagnostic_presentation_rows(rows: list[list[str]]) -> list[list[str]]:
        def category(check: str) -> str:
            if check in {"Ollama connectivity", "AI models"}:
                return "AI runtime"
            if check in {"Chat model", "Embedding model"}:
                return "Required models"
            if check == "Latest evaluation":
                return "Saved evaluation"
            if any(word in check.lower() for word in ("index", "manifest", "chroma")):
                return "Document index"
            return "Other"

        return [
            [category(row[0]), row[0], row[1], row[2]]
            for row in rows
        ]

    def preflight_ui(self):
        summary, rows, message_update, send_update = self.preflight()
        return (
            summary,
            self.diagnostic_presentation_rows(rows),
            message_update,
            send_update,
        )

    def load_ai_models_ui(self):
        summary, rows, message_update, send_update = self.load_ai_models()
        return (
            summary,
            self.diagnostic_presentation_rows(rows),
            message_update,
            send_update,
        )

    def refresh_workspace_state(
        self, filter_query: str | None = None, selected_document_id: str | None = None
    ):
        """Synchronize inexpensive workspace presentation state after each local action."""
        samples = self.document_samples(filter_query)
        error_rows = self.current_error_rows()
        readiness, system_status, message_update, send_update, load_ai_update = (
            self.preflight_shell_ui()
        )
        manifest = self.vector_db.manifest()
        record = manifest.documents.get(str(selected_document_id or ""))
        selected_is_visible = bool(
            record and any(str(row[0]) == record.relative_path for row in samples)
        )
        if selected_is_visible and record is not None:
            selection = (
                record.document_id,
                gr.update(
                    value=(
                        '<div class="selected-document">'
                        f'<strong>{escape(record.relative_path)}</strong>'
                        f'<span>{record.page_count} page(s) · '
                        f'{record.chunk_count} chunk(s)</span></div>'
                    ),
                    visible=True,
                ),
                gr.update(visible=True, interactive=True),
                "",
                gr.update(visible=False),
            )
        else:
            selection = self.reset_document_selection(samples)
        return (
            gr.update(samples=samples),
            self.corpus_summary_html(),
            gr.update(label=f"Indexing errors ({len(error_rows)})"),
            self.indexing_errors_html(error_rows),
            readiness,
            system_status,
            message_update,
            send_update,
            load_ai_update,
            *selection,
        )

    def filter_document_inventory(self, query: str | None):
        samples = self.document_samples(query)
        return gr.update(samples=samples), *self.reset_document_selection(samples)

    def index_selected_action_ui(
        self, files: list[str] | None, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        _documents, status, _errors, _readiness = self.index_selected(files, progress)
        return gr.update(value=status, visible=True)

    def reindex_changed_action_ui(
        self, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        _documents, status, _errors = self.reindex_changed(progress)
        return gr.update(value=status, visible=True)

    def rebuild_index_action_ui(
        self, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        _documents, status, _errors = self.rebuild_index(progress)
        return gr.update(value=status, visible=True)

    def delete_selected_action_ui(self, document_id: str | None):
        _documents, status, _errors, text, confirmation = self.delete_selected(
            document_id
        )
        return gr.update(value=status, visible=True), text, confirmation

    def index_selected_ui(
        self, files: list[str] | None, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        documents, status, errors, readiness = self.index_selected(files, progress)
        return (
            documents,
            gr.update(value=status, visible=True),
            gr.update(value=self.indexing_errors_html(errors), visible=bool(errors)),
            readiness,
        )

    def reindex_changed_ui(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        documents, status, errors = self.reindex_changed(progress)
        return (
            documents,
            gr.update(value=status, visible=True),
            gr.update(value=self.indexing_errors_html(errors), visible=bool(errors)),
        )

    def rebuild_index_ui(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        documents, status, errors = self.rebuild_index(progress)
        return (
            documents,
            gr.update(value=status, visible=True),
            gr.update(value=self.indexing_errors_html(errors), visible=bool(errors)),
        )

    def reconcile_manifest_index_ui(self):
        documents, status, errors = self.reconcile_manifest_index()
        return (
            documents,
            gr.update(value=status, visible=True),
            gr.update(value=self.indexing_errors_html(errors), visible=bool(errors)),
        )

    def delete_selected_ui(self, document_id: str | None):
        documents, status, errors, text, confirmation = self.delete_selected(document_id)
        return (
            documents,
            gr.update(value=status, visible=True),
            gr.update(value=self.indexing_errors_html(errors), visible=bool(errors)),
            text,
            confirmation,
        )

    def chat_ui(self, message: str, history: list[dict[str, str]], session_id: str):
        values = list(self.chat(message, history, session_id))
        values[3] = gr.update(value=values[3], visible=True)
        values[6] = self.scores_html(normalize_result_rows(values[6]))
        values[7] = self.trace_html(normalize_result_rows(values[7]))
        return tuple(values)

    def clear_ui(self, session_id: str):
        if self.rag_graph is not None:
            self.rag_graph.clear(session_id)
        return (
            [],
            {},
            gr.update(value="", visible=False),
            "Conversation cleared.",
            self.evidence_html({}),
            self.scores_html([]),
            self.trace_html([]),
        )

    def export_chat_ui(self, messages: list[Any], result: dict[str, Any]):
        return gr.update(
            value=self.export_chat(messages, result), visible=True, interactive=True
        )

    def load_latest_evaluation_ui(
        self,
        split: str = "development",
        systems: list[str] | str | None = None,
    ):
        metrics, failures, context, status = self.load_latest_evaluation()
        readiness = self.evaluation_readiness(split, systems or list(SYSTEMS))
        return self._evaluation_presentation_updates(
            metrics, failures, context, status, readiness=readiness
        )

    def initialize_evaluation_ui(
        self, split: str, systems: list[str] | str | None
    ):
        return self.load_latest_evaluation_ui(split, systems)

    def evaluation_options_ui(
        self, split: str, systems: list[str] | str | None
    ):
        readiness = self.evaluation_readiness(split, systems)
        has_result = readiness.latest_result is not None
        if readiness.state == "blocked":
            status = gr.update(
                value=render_status(
                    "warning", "Benchmark unavailable", readiness.problems[0]
                ),
                visible=True,
            )
        else:
            status = gr.update(value="", visible=False)
        return status, gr.update(
            value="Run new evaluation" if has_result else "Run standard benchmark",
            interactive=readiness.state != "blocked",
        )

    def run_evaluation_presentation_ui(self, split: str, systems: list[str] | str | None):
        metrics, failures, context, status = self.run_evaluation_ui(split, systems)
        readiness = self.evaluation_readiness(split, systems)
        return self._evaluation_presentation_updates(
            metrics, failures, context, status, readiness=readiness
        )

    @staticmethod
    def begin_evaluation_ui():
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(
                value=render_status(
                    "info",
                    "Running evaluation…",
                    "This may take several minutes for systems that use the local AI model.",
                ),
                visible=True,
            ),
        )

    def _evaluation_presentation_updates(
        self,
        metrics: Sequence[Sequence[Any]],
        failures: Sequence[Sequence[Any]],
        context: str,
        status: str,
        *,
        readiness: EvaluationReadiness | None = None,
    ):
        has_result = bool(metrics) and bool(context)
        blocked = readiness is not None and readiness.state == "blocked"
        if readiness is not None and readiness.state == "blocked" and not has_result:
            status = render_status("warning", "Benchmark unavailable", readiness.problems[0])
        return (
            gr.update(visible=has_result),
            gr.update(value=context, visible=has_result),
            gr.update(value=self.metrics_html(metrics), visible=has_result),
            self.failures_html(failures),
            gr.update(visible=has_result),
            gr.update(value=status, visible=True),
            gr.update(
                value="Run new evaluation" if has_result else "Run standard benchmark",
                interactive=not blocked,
            ),
            gr.update(interactive=True),
        )

    def preflight_presentation_ui(self):
        summary, rows, message_update, send_update = self.preflight_ui()
        return summary, self.system_status_html(rows), message_update, send_update

    def load_ai_models_presentation_ui(self):
        summary, rows, message_update, send_update = self.load_ai_models_ui()
        return summary, self.system_status_html(rows), message_update, send_update

    def preflight_shell_ui(self):
        summary, system_status, message_update, send_update = (
            self.preflight_presentation_ui()
        )
        return (
            summary,
            system_status,
            message_update,
            send_update,
            gr.update(visible=self.rag_graph is None),
        )

    def load_ai_models_shell_ui(self):
        summary, system_status, message_update, send_update = (
            self.load_ai_models_presentation_ui()
        )
        return (
            summary,
            system_status,
            message_update,
            send_update,
            gr.update(visible=self.rag_graph is None),
        )

    def create_interface(self) -> gr.Blocks:
        theme = Soft(primary_hue="indigo", neutral_hue="slate").set(
            body_background_fill="#080d18",
            body_text_color="#e7ecf6",
            background_fill_primary="#101827",
            background_fill_secondary="#172237",
            block_background_fill="#101827",
            block_border_color="#2b3952",
            block_label_background_fill="#101827",
            block_label_text_color="#a9b5c8",
            input_background_fill="#101827",
            input_border_color="#41516c",
            input_placeholder_color="#7e8ca4",
            button_secondary_background_fill="#172237",
            button_secondary_background_fill_hover="#1d2a42",
            button_secondary_border_color="#41516c",
            button_secondary_text_color="#e7ecf6",
        )
        initial_documents = self.document_samples()
        initial_errors = self.current_error_rows()
        with gr.Blocks(
            title="Local Document RAG",
            theme=theme,
            css_paths=APP_STYLESHEET,
            js=ACCESSIBILITY_BOOTSTRAP,
            fill_width=True,
        ) as interface:
            session_id = gr.State(lambda: str(uuid4()))
            latest_result = gr.State({})
            selected_document_id = gr.State("")
            gr.HTML(
                '<a class="skip-link" href="#chat-workspace">Skip to workspace</a>',
                elem_id="skip-navigation",
            )
            with gr.Row(elem_id="app-shell", elem_classes="shell-header"):
                gr.HTML(
                    """
                    <header class="app-header">
                      <div class="app-identity">
                        <span class="app-mark" aria-hidden="true">R</span>
                        <div>
                          <h1>Local Document RAG</h1>
                          <p>Private document questions with inspectable evidence.</p>
                        </div>
                      </div>
                    </header>
                    """,
                    elem_classes="shell-identity",
                )
                with gr.Row(elem_classes="shell-actions"):
                    readiness = gr.HTML(
                        render_status(
                            "warning",
                            "AI not loaded",
                            "Load the local models when you are ready to ask.",
                        ),
                        elem_id="readiness-status",
                        elem_classes=["readiness-summary", "status-host"],
                    )
                    load_ai = gr.Button(
                        "Load AI models",
                        variant="primary",
                        elem_classes=["primary-action", "load-models-action"],
                    )

            with gr.Tabs(elem_id="primary-tabs"):
                with gr.Tab("Workspace", elem_id="workspace-tab"):
                    with gr.Row(equal_height=False, elem_id="workspace-grid"):
                        with gr.Column(scale=4, min_width=0, elem_id="chat-workspace"):
                            gr.HTML(
                                """
                                <div class="workspace-heading">
                                  <div>
                                    <h2>Ask your documents</h2>
                                    <p>Answers stay connected to the evidence used.</p>
                                  </div>
                                </div>
                                """
                            )
                            answer_state = gr.HTML(
                                "",
                                visible=False,
                                elem_id="answer-status",
                                elem_classes=["answer-state", "status-host"],
                            )
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                type="messages",
                                allow_tags=False,
                                height=300,
                                elem_id="conversation-region",
                                elem_classes=[
                                    "conversation",
                                    "overflow-region",
                                    "fixed-scroll-region",
                                ],
                            )
                            message = gr.Textbox(
                                label="Question",
                                placeholder="Load AI models before asking about your documents",
                                interactive=False,
                                lines=2,
                                elem_classes="question-composer",
                            )
                            with gr.Row(elem_classes=["action-row", "chat-actions"]):
                                send = gr.Button(
                                    "Ask", variant="primary", interactive=False
                                )
                                clear = gr.Button("Clear")
                                export = gr.Button("Export")
                                export_file = gr.DownloadButton(
                                    "Download export",
                                    visible=False,
                                    interactive=False,
                                    elem_id="conversation-export",
                                )
                            gr.HTML(
                                """
                                <div class="section-heading evidence-heading">
                                  <div><h3>Cited sources</h3></div>
                                  <p>Only evidence cited by the answer appears here.</p>
                                </div>
                                """
                            )
                            sources = gr.HTML(
                                self.evidence_html({}),
                                elem_id="evidence-list",
                                elem_classes=["evidence-region", "overflow-region"],
                            )
                            with gr.Accordion(
                                "Technical details", open=False, elem_id="technical-details"
                            ):
                                evidence = gr.Markdown(
                                    "Routing and evidence details will appear after a question."
                                )
                                scores = gr.HTML(
                                    self.scores_html([]),
                                    elem_id="retrieval-scores-table",
                                    elem_classes=["retrieval-scores-table", "overflow-region"],
                                )
                                trace = gr.HTML(
                                    self.trace_html([]),
                                    elem_id="retrieval-trace-table",
                                    elem_classes=["retrieval-trace-table", "overflow-region"],
                                )

                        with gr.Column(
                            scale=1,
                            min_width=296,
                            elem_id="corpus-rail",
                        ):
                            gr.HTML(
                                """
                                <div class="documents-heading">
                                  <h2>Documents</h2>
                                  <p>Add, review, and maintain the local corpus.</p>
                                </div>
                                """
                            )
                            gr.HTML(
                                """
                                <div class="upload-intro">
                                  <strong>Add PDF or TXT</strong>
                                  <span>Drop files here or browse</span>
                                </div>
                                """
                            )
                            files = gr.File(
                                label="Add PDF or TXT",
                                show_label=False,
                                file_count="multiple",
                                file_types=[".pdf", ".txt"],
                                type="filepath",
                                elem_id="document-upload",
                                elem_classes="upload-compact",
                            )
                            index_button = gr.Button(
                                "Index files",
                                variant="primary",
                                elem_classes="primary-action",
                            )
                            ingestion_status = gr.HTML(
                                "",
                                visible=False,
                                elem_id="ingestion-status",
                                elem_classes=["inline-status", "status-host"],
                            )
                            corpus_summary = gr.HTML(
                                self.corpus_summary_html(),
                                elem_id="corpus-summary",
                            )
                            document_filter = gr.Textbox(
                                label="Filter documents",
                                placeholder="Filter by document or status",
                                lines=1,
                                elem_id="document-filter",
                            )
                            documents = gr.Dataset(
                                components=["textbox", "number", "number", "textbox"],
                                samples=initial_documents,
                                headers=DOCUMENT_HEADERS,
                                label="Indexed documents",
                                type="values",
                                layout="table",
                                samples_per_page=25,
                                elem_id="documents-table",
                                elem_classes=["documents-table", "overflow-region"],
                            )
                            selected_document = gr.HTML(
                                "",
                                visible=False,
                                elem_id="selected-document-summary",
                            )
                            delete_button = gr.Button(
                                "Delete selected",
                                visible=bool(initial_documents),
                                interactive=False,
                                elem_classes="destructive-review",
                            )
                            with gr.Group(
                                visible=False,
                                elem_id="deletion-alert",
                                elem_classes="delete-confirmation",
                            ) as delete_confirmation:
                                delete_confirmation_text = gr.HTML()
                                gr.Markdown(
                                    "This removes the document and its indexed chunks. "
                                    "This action cannot be undone."
                                )
                                with gr.Row(elem_classes="action-row"):
                                    cancel_delete = gr.Button("Cancel")
                                    confirm_delete = gr.Button(
                                        "Confirm deletion", variant="stop"
                                    )
                            with gr.Accordion(
                                "Maintenance", open=False, elem_id="maintenance-panel"
                            ):
                                gr.Markdown(
                                    "Reindex changed sources or rebuild the complete local index."
                                )
                                with gr.Row(elem_id="maintenance-actions"):
                                    reindex_button = gr.Button(
                                        "Reindex changed documents", variant="secondary"
                                    )
                                    rebuild_button = gr.Button(
                                        "Rebuild complete index",
                                        variant="stop",
                                        elem_classes="destructive-review",
                                    )
                            with gr.Accordion(
                                f"Indexing errors ({len(initial_errors)})",
                                open=False,
                                elem_id="index-errors-panel",
                            ) as index_errors_panel:
                                errors = gr.HTML(
                                    self.indexing_errors_html(initial_errors),
                                    elem_id="indexing-errors-table",
                                    elem_classes="indexing-errors-table",
                                )
                            with gr.Accordion(
                                "System status", open=False, elem_id="system-status-panel"
                            ):
                                system_status = gr.HTML(
                                    self.system_status_html([]),
                                    elem_id="system-status-details",
                                    elem_classes="system-status-details",
                                )

                with gr.Tab("Evaluation", elem_id="evaluation-tab"):
                    gr.HTML(
                        """
                        <div class="evaluation-heading">
                          <div>
                            <h2>Evaluation</h2>
                            <p>Compare retrieval and answer quality across the local RAG systems.</p>
                          </div>
                        </div>
                        """
                    )
                    with gr.Group(
                        visible=False,
                        elem_id="evaluation-results",
                    ) as evaluation_results:
                        result_context = gr.HTML(
                            "",
                            visible=False,
                            elem_id="evaluation-result-context",
                        )
                        metrics = gr.HTML(
                            self.metrics_html([]),
                            visible=False,
                            elem_id="evaluation-metrics-table",
                            elem_classes=["evaluation-metrics-table", "overflow-region"],
                        )
                        with gr.Accordion(
                            "Failure details",
                            open=False,
                            visible=False,
                            elem_id="evaluation-failures-panel",
                        ) as failure_panel:
                            failures = gr.HTML(
                                self.failures_html([]),
                                elem_id="evaluation-failures-table",
                                elem_classes=["evaluation-failures-table", "overflow-region"],
                            )
                    with gr.Group(elem_id="evaluation-workflow"):
                        with gr.Row(
                            elem_id="evaluation-actions",
                            elem_classes="evaluation-actions-row",
                        ):
                            run_eval = gr.Button(
                                "Run standard benchmark",
                                variant="primary",
                                elem_id="run-evaluation",
                            )
                            load_eval = gr.Button(
                                "Refresh latest result",
                                variant="secondary",
                                elem_id="load-evaluation",
                            )
                        eval_status = gr.HTML(
                            "",
                            visible=False,
                            elem_id="evaluation-status",
                            elem_classes=["inline-status", "status-host"],
                        )
                    with gr.Accordion(
                        "Advanced options",
                        open=False,
                        elem_id="evaluation-advanced-options",
                    ):
                        gr.Markdown(
                            "Standard: development split and all systems. "
                            "Agentic runs can take considerably longer."
                        )
                        with gr.Row(elem_classes="evaluation-config-row"):
                            split = gr.Dropdown(
                                [
                                    ("Development", "development"),
                                    ("Test — held-out final validation", "test"),
                                ],
                                value="development",
                                label="Split",
                                elem_id="evaluation-split",
                                min_width=200,
                                scale=0,
                            )
                            systems = gr.CheckboxGroup(
                                list(SYSTEMS),
                                value=list(SYSTEMS),
                                label="Systems",
                                elem_id="evaluation-systems",
                                min_width=320,
                                scale=1,
                            )
            selection_outputs = [
                selected_document_id,
                selected_document,
                delete_button,
                delete_confirmation_text,
                delete_confirmation,
            ]
            workspace_inputs = [document_filter, selected_document_id]
            workspace_outputs = [
                documents,
                corpus_summary,
                index_errors_panel,
                errors,
                readiness,
                system_status,
                message,
                send,
                load_ai,
                *selection_outputs,
            ]
            index_event = index_button.click(
                self.index_selected_action_ui,
                files,
                ingestion_status,
            )
            index_event.then(
                self.refresh_workspace_state, workspace_inputs, workspace_outputs
            )
            reindex_event = reindex_button.click(
                self.reindex_changed_action_ui,
                None,
                ingestion_status,
            )
            reindex_event.then(
                self.refresh_workspace_state, workspace_inputs, workspace_outputs
            )
            rebuild_event = rebuild_button.click(
                self.rebuild_index_action_ui,
                None,
                ingestion_status,
            )
            rebuild_event.then(
                self.refresh_workspace_state, workspace_inputs, workspace_outputs
            )

            documents.select(
                self.select_document,
                document_filter,
                selection_outputs,
            )
            document_filter.input(
                self.filter_document_inventory,
                document_filter,
                [documents, *selection_outputs],
                show_progress="hidden",
            )
            delete_button.click(
                self.prepare_deletion,
                selected_document_id,
                [delete_confirmation_text, delete_confirmation],
            )
            cancel_delete.click(
                self.cancel_deletion,
                None,
                [delete_confirmation_text, delete_confirmation],
            )
            delete_event = confirm_delete.click(
                self.delete_selected_action_ui,
                selected_document_id,
                [
                    ingestion_status,
                    delete_confirmation_text,
                    delete_confirmation,
                ],
            )
            delete_event.then(
                self.refresh_workspace_state, workspace_inputs, workspace_outputs
            )

            for trigger in (send.click, message.submit):
                trigger(
                    self.chat_ui,
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
                self.clear_ui,
                session_id,
                [chatbot, latest_result, answer_state, evidence, sources, scores, trace],
            )
            export.click(self.export_chat_ui, [chatbot, latest_result], export_file)
            evaluation_outputs = [
                evaluation_results,
                result_context,
                metrics,
                failures,
                failure_panel,
                eval_status,
                run_eval,
                load_eval,
            ]
            run_evaluation_event = run_eval.click(
                self.begin_evaluation_ui,
                None,
                [run_eval, load_eval, eval_status],
                queue=False,
            ).then(
                self.run_evaluation_presentation_ui,
                [split, systems],
                evaluation_outputs,
            )
            run_evaluation_event.then(
                self.refresh_workspace_state, workspace_inputs, workspace_outputs
            )
            load_evaluation_event = load_eval.click(
                self.begin_evaluation_ui,
                None,
                [run_eval, load_eval, eval_status],
                queue=False,
            ).then(
                self.load_latest_evaluation_ui,
                [split, systems],
                evaluation_outputs,
            )
            load_evaluation_event.then(
                self.refresh_workspace_state, workspace_inputs, workspace_outputs
            )
            for option_event in (split.change, systems.change):
                option_event(
                    self.evaluation_options_ui,
                    [split, systems],
                    [eval_status, run_eval],
                    show_progress="hidden",
                )
            preflight_outputs = [readiness, system_status, message, send, load_ai]
            load_models_event = load_ai.click(
                self.load_ai_models_shell_ui, None, preflight_outputs
            )
            load_models_event.then(
                self.refresh_workspace_state, workspace_inputs, workspace_outputs
            )
            interface.load(
                self.refresh_workspace_state,
                workspace_inputs,
                workspace_outputs,
                show_progress="hidden",
            )
            interface.load(
                self.initialize_evaluation_ui,
                [split, systems],
                evaluation_outputs,
                show_progress="hidden",
            )
        return interface


app = RAGApplication()


def main() -> int:
    app.create_interface().launch(
        server_name=config.gradio_host, server_port=config.gradio_port, share=config.gradio_share
    )
    return 0


if __name__ == "__main__":
    main()
