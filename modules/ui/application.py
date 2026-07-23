"""Gradio interface exposing the existing local RAG services."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from html import escape
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import gradio as gr

from modules.config import PROJECT_ROOT, config
from modules.contracts import ApplicationVectorStore, GraphVectorStore, TableRow
from modules.evaluation import (
    SYSTEMS,
    Split,
    SystemName,
    evaluation_result_kind,
    is_standard_benchmark_summary,
    load_cases,
    normalize_model_name,
    preflight_multihop,
    required_models_for_systems,
    run_evaluation,
)
from modules.rag_graph import RAGGraph
from modules.ui.contracts import (
    AttentionItem,
    CheckState,
    CorpusSnapshot,
    DashboardSnapshot,
    EvaluationCategorySummary,
    EvaluationMetricObservation,
    EvaluationMetricSummary,
    EvaluationPageSnapshot,
    EvaluationSummary,
    IndexSnapshot,
    QueryDiagnostics,
    QuerySnapshot,
    RetrievalHitView,
    RuntimeSnapshot,
    SafeConfigurationValue,
    SourceView,
    SystemCheck,
    SystemPageSnapshot,
    SystemPageState,
    TraceEventView,
)
from modules.ui.events import component_update
from modules.ui.presenters import (
    render_corpus_summary,
    render_evidence,
    render_indexing_errors,
    render_result_table,
    render_selected_document,
    render_status,
)
from modules.vector_db import VectorDBManager

DOCUMENT_HEADERS = ["Document", "Pages", "Chunks", "Status"]
TRACE_HEADERS = [
    "Stage",
    "Decision",
    "Retrieved",
    "Fused",
    "Selected",
    "Retry",
    "LLM calls",
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


def _evaluation_systems(configuration: Mapping[str, object]) -> tuple[str, ...]:
    systems = configuration.get("systems", [])
    if not isinstance(systems, list):
        return ()
    return tuple(str(system) for system in systems)


def _metric_observation(
    *,
    name: str,
    system: str,
    system_metrics: Mapping[str, object],
) -> EvaluationMetricObservation | None:
    metrics = system_metrics.get(system)
    if not isinstance(metrics, dict):
        return None
    observation = metrics.get(name)
    if not isinstance(observation, dict):
        return None
    raw_value = observation.get("value")
    sample_count = observation.get("sample_count")
    return EvaluationMetricObservation(
        system=system,
        value=float(raw_value) if isinstance(raw_value, (int, float)) else None,
        status=str(observation.get("status") or "missing"),
        sample_count=sample_count if isinstance(sample_count, int) else None,
    )


def _quality_categories(
    *,
    systems: tuple[str, ...],
    system_metrics: Mapping[str, object],
) -> tuple[EvaluationCategorySummary, ...]:
    categories = []
    for category, names in METRIC_GROUPS:
        metrics = []
        for name in names:
            observations = tuple(
                observation
                for system in systems
                if (
                    observation := _metric_observation(
                        name=name,
                        system=system,
                        system_metrics=system_metrics,
                    )
                )
                is not None
            )
            if observations:
                metrics.append(
                    EvaluationMetricSummary(
                        name=name,
                        label=DISPLAY_METRIC_LABELS[name],
                        observations=observations,
                    )
                )
        categories.append(EvaluationCategorySummary(category, tuple(metrics)))
    return tuple(categories)


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
    chat_model: str = ""
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


def _mapping_rows(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _coerce_int(value: object) -> int:
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _answer_state(
    result: Mapping[str, object],
    *,
    override: str | None,
) -> str:
    if override is not None:
        return override
    evidence = str(result.get("evidence_status") or "")
    if evidence == "sufficient":
        return "supported"
    if evidence in {"limited", "insufficient"}:
        return "limited" if evidence == "limited" else "abstention"
    termination = next(
        (
            str(event.get("termination"))
            for event in reversed(_mapping_rows(result.get("trace")))
            if event.get("termination")
        ),
        "",
    )
    return {"supported": "supported", "limited": "limited"}.get(termination, "completed")


def format_metric(name: str, value: Any) -> str:
    if value is None or value == "":
        return "—"
    if name in PERCENTAGE_METRICS or (name not in METRIC_NAMES and 0 <= float(value) <= 1):
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
        raise ValueError("Evaluation summaries must use schema version 2. Run a new evaluation.")
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


def normalize_result_rows(value: Any) -> list[list[Any]]:
    """Narrow Gradio's broad component value type to safe tabular rows."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [
        list(row)
        for row in value
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
    ]


ACCESSIBILITY_BOOTSTRAP = Path(__file__).with_name("accessibility.js").read_text(encoding="utf-8")


class RAGApplication:
    def __init__(self, vector_db: ApplicationVectorStore | None = None):
        self.vector_db = vector_db or VectorDBManager()
        self.rag_graph: RAGGraph | None = None
        self.last_errors: dict[str, str] = {}

    def initialize(self) -> None:
        self.vector_db.setup()
        self.rag_graph = RAGGraph(cast(GraphVectorStore, self.vector_db))

    def _graph(self) -> RAGGraph:
        if self.rag_graph is None:
            self.initialize()
        assert self.rag_graph is not None
        return self.rag_graph

    @staticmethod
    def _check_area(name: str) -> str:
        if name in {"Ollama connectivity", "AI models"}:
            return "runtime"
        if name in {"Chat model", "Embedding model"}:
            return "models"
        if name == "Latest evaluation":
            return "evaluation"
        return "index"

    def runtime_snapshot(self) -> RuntimeSnapshot:
        """Return local runtime readiness without presentation markup."""
        rows = self.diagnostic_rows()
        runtime_rows = rows[:3] + rows[-2:-1]
        state_map: dict[str, CheckState] = {
            "ok": "ready",
            "warning": "review",
            "error": "blocked",
            "pending": "not_loaded",
        }
        checks = tuple(
            SystemCheck(
                area=self._check_area(name),
                name=name,
                state=cast(Any, state_map.get(state, "error")),
                detail=detail,
            )
            for name, state, detail in runtime_rows
        )
        blocked = [check for check in checks if check.state in {"blocked", "error"}]
        if blocked:
            actions = []
            if any(check.name == "Ollama connectivity" for check in blocked):
                actions.append("Start Ollama with `ollama serve`.")
            for label, model in (
                ("Chat model", config.llm_model),
                ("Embedding model", config.embedding_model),
            ):
                if any(check.name == label for check in blocked):
                    actions.append(f"Install the model with `ollama pull {model}`.")
            return RuntimeSnapshot(
                state="blocked",
                title="Not ready for questions",
                detail=" ".join(actions),
                chat_enabled=False,
                can_load_models=False,
                checks=checks,
                chat_model=config.llm_model,
                embedding_model=config.embedding_model,
            )
        loaded = self.rag_graph is not None
        return RuntimeSnapshot(
            state="ready" if loaded else "not_loaded",
            title="Ready for questions" if loaded else "Application ready; AI not loaded",
            detail=(
                "Ollama, required models, index, and graph are available."
                if loaded
                else "Load AI models to enable document questions."
            ),
            chat_enabled=loaded,
            can_load_models=not loaded,
            checks=checks,
            chat_model=config.llm_model,
            embedding_model=config.embedding_model,
        )

    def corpus_snapshot(self) -> CorpusSnapshot:
        """Return corpus totals without document identifiers or markup."""
        try:
            records = tuple(self.vector_db.manifest().documents.values())
            document_count = len(records)
            return CorpusSnapshot(
                document_count=document_count,
                page_count=sum(record.page_count for record in records),
                chunk_count=sum(record.chunk_count for record in records),
                status=("review" if self.last_errors else ("ready" if document_count else "empty")),
            )
        except Exception:
            return CorpusSnapshot(0, 0, 0, "error")

    def index_snapshot(self) -> IndexSnapshot:
        """Return reconciliation counts without exposing stable IDs."""
        try:
            result = self.vector_db.reconcile_index()
            missing = len(result.missing_chunk_ids)
            orphan = len(result.orphan_chunk_ids)
            duplicates = len(result.duplicate_chunk_ids)
            missing_files = len(result.missing_source_files)
            incompatible = len(result.incompatible_document_ids)
            state = (
                "error"
                if missing or duplicates or incompatible
                else ("review" if orphan or missing_files else "ready")
            )
            return IndexSnapshot(
                missing,
                orphan,
                duplicates,
                missing_files,
                incompatible,
                state,
            )
        except Exception:
            return IndexSnapshot(0, 0, 0, 0, 0, "error")

    @staticmethod
    def _evaluation_summary(path: Path | None) -> EvaluationSummary | None:
        if path is None:
            return None
        try:
            summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            configuration = summary.get("configuration", {})
            if not isinstance(configuration, dict):
                configuration = {}
            systems = _evaluation_systems(configuration)
            system_metrics = summary.get("metrics", {})
            if not isinstance(system_metrics, dict):
                system_metrics = {}
            cases = sum(
                bool(line.strip())
                for line in (path / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            )
            result_kind = (
                "standard" if evaluation_result_kind(summary) == "standard_benchmark" else "custom"
            )
            return EvaluationSummary(
                result_path=str(path),
                split=str(configuration.get("evaluated_split") or "—"),
                systems=systems,
                case_count=cases,
                result_kind=result_kind,
                created_at=str(
                    configuration.get("timestamp")
                    or datetime.fromtimestamp(
                        (path / "summary.json").stat().st_mtime, UTC
                    ).isoformat()
                ),
                chat_model=str(configuration.get("chat_model") or "—"),
                quality_categories=_quality_categories(
                    systems=systems,
                    system_metrics=system_metrics,
                ),
            )
        except (OSError, json.JSONDecodeError):
            return None

    def dashboard_snapshot(self) -> DashboardSnapshot:
        """Compose the current overview state from plain snapshots."""
        runtime = self.runtime_snapshot()
        corpus = self.corpus_snapshot()
        index = self.index_snapshot()
        attention: list[AttentionItem] = []
        if runtime.state in {"blocked", "error"}:
            attention.append(AttentionItem("runtime", runtime.detail, "error"))
        if corpus.status in {"review", "error"}:
            attention.append(AttentionItem("corpus", "Review corpus indexing errors.", "warning"))
        if index.status != "ready":
            attention.append(
                AttentionItem(
                    "index",
                    "Review manifest and Chroma reconciliation.",
                    "error" if index.status == "error" else "warning",
                )
            )
        return DashboardSnapshot(
            runtime=runtime,
            corpus=corpus,
            index=index,
            evaluation=self.latest_evaluation_summary(),
            attention_items=tuple(attention),
        )

    def latest_evaluation_summary(self) -> EvaluationSummary | None:
        """Return the latest compatible standard result as presentation-neutral data."""
        return self._evaluation_summary(self.latest_evaluation())

    def _system_check(self, *, row: Sequence[str], name: str | None = None) -> SystemCheck:
        state_map: dict[str, CheckState] = {
            "ok": "ready",
            "warning": "review",
            "error": "blocked",
            "pending": "not_loaded",
        }
        return SystemCheck(
            area=self._check_area(row[0]),
            name=name or row[0],
            state=state_map.get(row[1], "error"),
            detail=row[2],
        )

    def _system_runtime_checks(
        self,
        *,
        rows: Sequence[Sequence[str]],
    ) -> tuple[SystemCheck, ...]:
        by_name = {row[0]: row for row in rows}
        return tuple(
            self._system_check(
                row=by_name[name],
                name="AI service initialization" if name == "AI models" else name,
            )
            for name in ("Ollama connectivity", "Chat model", "Embedding model", "AI models")
            if name in by_name
        )

    def _system_index_checks(
        self,
        *,
        rows: Sequence[Sequence[str]],
    ) -> tuple[SystemCheck, ...]:
        by_name = {row[0]: row for row in rows}
        index_labels = (
            ("Chroma collection", "Chroma collection"),
            ("Manifest", "Manifest"),
            ("Missing Chroma chunks", "Missing chunks"),
            ("Orphan Chroma chunks", "Orphan chunks"),
            ("Duplicate IDs", "Duplicate IDs"),
            ("Missing source files", "Missing source files"),
            ("Index configuration", "Index compatibility"),
        )
        index_checks = tuple(
            self._system_check(row=by_name[source], name=label)
            for source, label in index_labels
            if source in by_name
        )
        if not index_checks and "Index diagnostics" in by_name:
            index_checks = (
                SystemCheck(
                    "index",
                    "Index diagnostics",
                    "error",
                    "Index diagnostics are unavailable.",
                ),
            )
        return index_checks

    def _system_evaluation_checks(
        self,
        *,
        readiness: EvaluationReadiness,
        runtime_checks: Sequence[SystemCheck],
    ) -> tuple[SystemCheck, ...]:
        dataset_problem = self._system_dataset_problem(problems=readiness.problems)
        required_models = required_models_for_systems(
            cast(Sequence[SystemName], EVALUATION_SYSTEMS)
        )
        model_checks = {item.name: item for item in runtime_checks if item.name.endswith("model")}
        models_ready = all(item.state == "ready" for item in model_checks.values())
        return (
            SystemCheck(
                "evaluation",
                "Dataset readiness",
                "blocked" if dataset_problem else "ready",
                dataset_problem
                or "Development and held-out benchmark cases are prepared and indexed.",
            ),
            SystemCheck(
                "evaluation",
                "Required models",
                "ready" if models_ready else "blocked",
                ", ".join(required_models),
            ),
            SystemCheck(
                "evaluation",
                "Latest compatible standard result",
                "ready" if readiness.latest_result else "not_loaded",
                (
                    "Compatible schema-v2 standard benchmark available."
                    if readiness.latest_result
                    else "No stored result"
                ),
            ),
        )

    @staticmethod
    def _system_dataset_problem(*, problems: Sequence[str]) -> str | None:
        markers = (
            "multihop",
            "benchmark",
            "development",
            "held-out",
            "preparation script",
        )
        return next(
            (
                problem
                for problem in problems
                if any(marker in problem.casefold() for marker in markers)
            ),
            None,
        )

    @staticmethod
    def _system_summary(
        *,
        checks: Sequence[SystemCheck],
    ) -> tuple[SystemPageState, str, str, bool]:
        blocking = tuple(item for item in checks if item.state in {"blocked", "error"})
        reviewing = tuple(item for item in checks if item.state in {"review", "not_loaded"})
        ollama_blocked = any(item.name == "Ollama connectivity" for item in blocking)
        missing_models = tuple(
            item for item in blocking if item.name in {"Chat model", "Embedding model"}
        )
        if blocking:
            title = "Unavailable"
            if ollama_blocked:
                detail = f"Start Ollama at {config.ollama_base_url}."
            elif missing_models:
                detail = " ".join(
                    f"Run `ollama pull {config.llm_model if item.name == 'Chat model' else config.embedding_model}`."
                    for item in missing_models
                )
            else:
                detail = "Review the blocked checks before using dependent workflows."
            return "blocked", title, detail, False
        if reviewing:
            can_load_models = any(item.name == "AI service initialization" for item in reviewing)
            return (
                "review",
                "Review",
                "Some local services or stored results need attention.",
                can_load_models,
            )
        return (
            "ready",
            "Ready",
            "Ollama, required models, index, and application services are available.",
            False,
        )

    @staticmethod
    def _system_safe_configuration() -> tuple[SafeConfigurationValue, ...]:
        return (
            SafeConfigurationValue("Ollama base URL", config.ollama_base_url),
            SafeConfigurationValue("Chat model", config.llm_model),
            SafeConfigurationValue("Embedding model", config.embedding_model),
            SafeConfigurationValue(
                "Gradio",
                f"{config.gradio_host}:{config.gradio_port}",
            ),
            SafeConfigurationValue(
                "Chunk size / overlap",
                f"{config.chunk_size} / {config.chunk_overlap}",
            ),
            SafeConfigurationValue(
                "Retrieval candidates",
                (
                    f"semantic {config.semantic_candidates} · "
                    f"sparse {config.sparse_candidates} · max {config.max_candidates}"
                ),
            ),
            SafeConfigurationValue(
                "Context / subqueries / retries",
                (f"{config.max_context_chunks} / {config.max_subqueries} / {config.max_retries}"),
            ),
        )

    def system_snapshot(self) -> SystemPageSnapshot:
        """Return sanitized runtime, index, evaluation, and configuration diagnostics."""
        ollama = self._ollama_info()
        rows = self.diagnostic_rows(ollama=ollama)
        runtime_checks = self._system_runtime_checks(rows=rows)
        index_checks = self._system_index_checks(rows=rows)
        readiness = self.evaluation_readiness(
            "development",
            EVALUATION_SYSTEMS,
            ollama=ollama,
        )
        evaluation_checks = self._system_evaluation_checks(
            readiness=readiness,
            runtime_checks=runtime_checks,
        )
        state, title, detail, can_load_models = self._system_summary(
            checks=runtime_checks + index_checks + evaluation_checks
        )
        return SystemPageSnapshot(
            state=state,
            title=title,
            detail=detail,
            can_load_models=can_load_models,
            runtime_checks=runtime_checks,
            index_checks=index_checks,
            evaluation_checks=evaluation_checks,
            safe_configuration=self._system_safe_configuration(),
        )

    def query_snapshot(self, message: str, session_id: str) -> QuerySnapshot:
        """Run a query and expose only public answer and observability data."""
        if not message.strip():
            return self._query_contract({}, answer="", state="completed")
        try:
            result = self._graph().process_query(message.strip(), session_id)
        except Exception:
            return self._query_contract(
                {},
                answer="The local RAG service is unavailable. Review System status and retry.",
                state="unavailable",
            )
        return self._query_contract(result)

    @staticmethod
    def _query_contract(
        result: Mapping[str, object],
        *,
        answer: str | None = None,
        state: str | None = None,
    ) -> QuerySnapshot:
        evidence = str(result.get("evidence_status") or "")
        validation = result.get("validation")
        validation_data: Mapping[str, object] = validation if isinstance(validation, dict) else {}
        return QuerySnapshot(
            answer=answer if answer is not None else str(result.get("answer") or ""),
            answer_state=cast(Any, _answer_state(result, override=state)),
            sources=tuple(
                SourceView(
                    label=str(source.get("label") or ""),
                    filename=str(source.get("filename") or ""),
                    page=_optional_int(source.get("page")),
                    excerpt=str(source.get("excerpt") or ""),
                )
                for source in _mapping_rows(result.get("sources"))
            ),
            retrieval_hits=tuple(
                RetrievalHitView(
                    chunk_id=str(hit.get("chunk_id") or ""),
                    filename=str(hit.get("filename") or ""),
                    page=_optional_int(hit.get("page")),
                    semantic_score=_optional_float(hit.get("semantic_score")),
                    sparse_score=_optional_float(hit.get("sparse_score")),
                    fused_score=_optional_float(hit.get("fused_score")),
                    selection_score=_optional_float(hit.get("selection_score")),
                    matched_subqueries=_string_tuple(hit.get("subqueries")),
                )
                for hit in _mapping_rows(result.get("retrieval_hits"))
            ),
            trace=tuple(
                TraceEventView(
                    stage=str(event.get("stage") or ""),
                    decision=str(event.get("decision") or ""),
                    retrieved_count=_optional_int(
                        event.get("retrieved_count", event.get("candidate_count"))
                    ),
                    fused_count=_optional_int(event.get("fused_count")),
                    selected_count=_optional_int(event.get("selected_count")),
                    retry_count=_coerce_int(event.get("retry_count")),
                    llm_calls=_coerce_int(event.get("llm_calls")),
                    termination=str(event.get("termination") or ""),
                    duration_ms=_optional_float(event.get("duration_ms")),
                )
                for event in _mapping_rows(result.get("trace"))
            ),
            diagnostics=QueryDiagnostics(
                route=str(result.get("route") or ""),
                retrieval_strategy=str(result.get("strategy") or ""),
                subqueries=_string_tuple(result.get("subqueries")),
                retry_count=_coerce_int(result.get("retry_count")),
                evidence_state=evidence,
                conflict_state="conflict" if result.get("conflict") else "none",
                citation_validation=(
                    "valid"
                    if validation_data.get("is_valid") is True
                    else ("invalid" if validation_data else "not_reported")
                ),
            ),
        )

    def evaluation_snapshot(
        self,
        split: str,
        systems: Sequence[str],
        chat_model: str | None = None,
    ) -> EvaluationPageSnapshot:
        """Return evaluation readiness and the latest compatible standard result."""
        readiness = self.evaluation_readiness(split, systems, chat_model)
        return self._evaluation_snapshot_from_result(
            readiness=readiness,
            result_path=readiness.latest_result,
        )

    def _evaluation_snapshot_from_result(
        self,
        *,
        readiness: EvaluationReadiness,
        result_path: Path | None,
        problems: tuple[str, ...] | None = None,
        state: EvaluationState | None = None,
    ) -> EvaluationPageSnapshot:
        latest = self._evaluation_summary(result_path)
        metric_rows: tuple[tuple[str, ...], ...] = ()
        failure_rows: tuple[tuple[str, ...], ...] = ()
        if result_path is not None and latest is not None:
            metrics, failures, _context, _status = self.load_evaluation_result(result_path)
            metric_rows = tuple(tuple(str(value) for value in row) for row in metrics)
            failure_rows = tuple(tuple(str(value) for value in row) for row in failures)
        return EvaluationPageSnapshot(
            state=(
                "blocked"
                if (state or readiness.state) == "blocked"
                else (
                    "error"
                    if (state or readiness.state) == "error"
                    else ("saved_result" if latest is not None else "ready")
                )
            ),
            split=readiness.split,
            systems=readiness.systems,
            requires_index=readiness.requires_index,
            requires_embeddings=readiness.requires_embeddings,
            requires_chat=readiness.requires_chat,
            problems=readiness.problems if problems is None else problems,
            latest=latest,
            chat_model=readiness.chat_model,
            metric_rows=metric_rows,
            failure_rows=failure_rows,
        )

    def run_evaluation_snapshot(
        self,
        *,
        split: str,
        systems: Sequence[str] | str | None,
        chat_model: str | None = None,
    ) -> EvaluationPageSnapshot:
        """Run an evaluation and return the exact result produced by that run."""
        readiness = self.evaluation_readiness(split, systems, chat_model)
        if readiness.state == "blocked":
            return self._evaluation_snapshot_from_result(
                readiness=readiness,
                result_path=readiness.latest_result,
            )
        dataset = PROJECT_ROOT / "evals" / "multihop" / "cases.jsonl"
        try:
            output = run_evaluation(
                dataset,
                cast(list[SystemName], list(readiness.systems)),
                cast(Split, split),
                dataset_name="multihop",
                chat_model=readiness.chat_model,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return self._evaluation_snapshot_from_result(
                readiness=readiness,
                result_path=readiness.latest_result,
                problems=(str(exc),),
                state="error",
            )
        completed = EvaluationReadiness(
            state="result",
            latest_result=output,
            systems=readiness.systems,
            split=readiness.split,
            requires_index=readiness.requires_index,
            requires_embeddings=readiness.requires_embeddings,
            requires_chat=readiness.requires_chat,
            chat_model=readiness.chat_model,
        )
        return self._evaluation_snapshot_from_result(
            readiness=completed,
            result_path=output,
        )

    def document_rows(self) -> list[TableRow]:
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

    def document_samples(self, query: str | None = None) -> list[TableRow]:
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
        return render_corpus_summary(self.corpus_snapshot())

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
        summary = render_selected_document(
            relative_path=record.relative_path,
            page_count=record.page_count,
            chunk_count=record.chunk_count,
        )
        return (
            record.document_id,
            component_update(value=summary, visible=True),
            component_update(visible=True, interactive=True),
            "",
            component_update(visible=False),
        )

    def reset_document_selection(self, rows: Any = None):
        normalized = self.document_rows() if rows is None else normalize_result_rows(rows)
        return (
            "",
            component_update(value="", visible=False),
            component_update(visible=bool(normalized), interactive=False),
            "",
            component_update(visible=False),
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
        return render_indexing_errors(rows)

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
                    f'<span class="metric-support">{escape(support)}</span>' if support else ""
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
            "<caption>Metrics comparison</caption>"
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
                "<strong>Status unknown</strong>"
                "<span>Workspace checks have not run yet.</span></div>"
                '<h4 class="system-status__technical-title">Technical values</h4>'
                f"{technical}</section>"
            )
        problem_rows = [row for row in normalized if len(row) >= 4 and str(row[2]) != "Ready"]
        if problem_rows:
            has_error = any(
                str(row[2]).casefold() in {"error", "unavailable"} for row in problem_rows
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
                overview.append(f"<section><h4>{escape(category)}</h4><ul>")
                for item in items:
                    overview.append(
                        f"<li><strong>{escape(str(item[1]))}</strong>"
                        f"<span>{escape(str(item[3]))}</span></li>"
                    )
                overview.append("</ul></section>")
            overview.append("</div>")
            summary = "".join(overview)
        else:
            summary = (
                '<div class="system-status__summary system-status__summary--ready">'
                "<strong>System ready</strong>"
                "<span>Local services, models, and indexed data are available.</span></div>"
            )
        technical = render_result_table(
            ["Area", "Check", "Status", "Details"],
            normalized,
            caption="Technical system values",
            empty_message="Workspace checks have not run yet.",
            mobile_cards=True,
            table_class="system-status-view",
        )
        role = (
            "alert"
            if any(str(row[2]).casefold() in {"error", "unavailable"} for row in problem_rows)
            else "status"
        )
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
                component_update(visible=False),
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
            component_update(visible=False),
        )

    def prepare_deletion(self, document_id: str | None):
        if not document_id:
            return "", component_update(visible=False)
        record = self.vector_db.manifest().documents.get(document_id)
        filename = record.relative_path if record else "selected document"
        return (
            f"<strong>Delete {escape(filename)}</strong> and its indexed chunks?",
            component_update(visible=True),
        )

    @staticmethod
    def cancel_deletion():
        return "", component_update(visible=False)

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
            status = render_status("error", "Reconciliation failed", f"{type(exc).__name__}: {exc}")
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

    def evaluation_model_choices(self, *, ollama: dict[str, Any] | None = None) -> tuple[str, ...]:
        """Return normalized local chat tags with the configured default first."""
        info = ollama or self._ollama_info()
        configured = normalize_model_name(config.llm_model)
        embedding = normalize_model_name(config.embedding_model)
        installed = (
            normalize_model_name(str(name)) for name in info.get("models", []) if str(name).strip()
        )
        return tuple(
            dict.fromkeys([configured, *(name for name in installed if name != embedding)])
        )

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
            summary = render_status("error", "Not ready for questions", "; ".join(actions) + ".")
        return (
            summary,
            self.diagnostic_display_rows(rows),
            component_update(interactive=ready),
            component_update(interactive=ready),
        )

    def load_ai_models(self, ollama: dict[str, Any] | None = None):
        """Initialize AI-backed services only after an explicit UI action."""
        rows = self.diagnostic_rows(ollama=ollama)
        if any(row[1] == "error" for row in rows):
            _, display_rows, message_update, send_update = self.preflight(ollama=ollama)
            return (
                render_status("error", "AI models not loaded", "Open System status, then retry."),
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
                render_status("error", "AI models not loaded", "Open System status, then retry."),
                self.diagnostic_display_rows(rows),
                component_update(interactive=False),
                component_update(interactive=False),
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
        return render_evidence(result.get("sources", []))

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
                event.get("llm_calls", 0),
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
        return render_status("info", "Completed", "Review the evidence and citations below.")

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
        systems_text = (
            ", ".join(system_labels.get(system, readable_label(system)) for system in systems)
            or "—"
        )
        raw_timestamp = configuration.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
            timestamp = timestamp.astimezone(UTC)
        except (TypeError, ValueError):
            timestamp = datetime.fromtimestamp((path / "summary.json").stat().st_mtime, UTC)
        case_label = f"{case_count} case" if case_count == 1 else f"{case_count} cases"
        result_kind = evaluation_result_kind(summary)
        result_label = (
            "Standard benchmark" if result_kind == "standard_benchmark" else "Custom evaluation"
        )
        items = (
            ("Type", result_label),
            ("Run", run_id),
            ("Dataset", f"{dataset} · {split}"),
            ("Systems", systems_text),
            ("Model", configuration.get("chat_model") or "—"),
            ("Cases", case_label),
            ("Result date", timestamp.strftime("%Y-%m-%d %H:%M UTC")),
        )
        content = "".join(
            '<div class="evaluation-context__item">'
            f"<span>{escape(str(label))}</span><strong>{escape(str(value))}</strong>"
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
        available_names = {name for values in system_metrics.values() for name in values}
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
        result_label = (
            "Standard benchmark"
            if evaluation_result_kind(summary) == "standard_benchmark"
            else "Custom evaluation"
        )
        context = RAGApplication.evaluation_context_html(path, summary, len(case_lines))
        return (
            metrics,
            failures,
            context,
            render_status("success", f"{result_label} loaded", f"{run_id} · {path}"),
        )

    @staticmethod
    def latest_evaluation() -> Path | None:
        root = PROJECT_ROOT / "evals" / "results" / "multihop"
        candidates: list[tuple[float, Path]] = []
        for summary_path in root.rglob("summary.json"):
            result_path = summary_path.parent
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not is_standard_benchmark_summary(summary):
                    continue
                if not (result_path / "cases.jsonl").is_file():
                    continue
                for line in (result_path / "cases.jsonl").read_text(encoding="utf-8").splitlines():
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
                    "No standard benchmark available",
                    "Run the standard benchmark to create a complete schema version 2 result.",
                ),
            )
        )

    def evaluation_readiness(  # lanorme: ignore[SIZE-002,COMPLEXITY-001,KWARG-001] -- One cohesive preflight
        self,
        split: str,
        systems: Sequence[str] | str | None,
        chat_model: str | None = None,
        *,
        ollama: dict[str, Any] | None = None,
    ) -> EvaluationReadiness:
        requested = [systems] if isinstance(systems, str) else list(systems or [])
        selected = tuple(system for system in requested if system in SYSTEMS)
        selected_chat_model = normalize_model_name(chat_model or config.llm_model)
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
        required = required_models_for_systems(
            cast(tuple[SystemName, ...], selected),
            selected_chat_model,
        )
        if required:
            info = ollama or self._ollama_info()
            if not info.get("reachable"):
                problems.append("Start Ollama with: ollama serve")
            else:
                available = {normalize_model_name(name) for name in info.get("models", [])}
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
            chat_model=selected_chat_model,
            problems=tuple(dict.fromkeys(problems)),
        )

    def run_evaluation_ui(  # lanorme: ignore[KWARG-001,TYPE-004] -- Legacy positional adapter
        self,
        split: str,
        systems: list[str] | str | None,
        chat_model: str | None = None,
    ):
        if not systems:
            return (
                [],
                [],
                "",
                render_status(
                    "warning", "No systems selected", "Select at least one evaluation system."
                ),
            )
        requested = [systems] if isinstance(systems, str) else list(systems)
        if not requested:
            return (
                [],
                [],
                "",
                render_status(
                    "warning", "No systems selected", "Select at least one evaluation system."
                ),
            )
        dataset = PROJECT_ROOT / "evals" / "multihop" / "cases.jsonl"
        selected = cast(list[SystemName], requested)
        evaluated_split = cast(Split, split)
        try:
            output = run_evaluation(
                dataset,
                selected,
                evaluated_split,
                dataset_name="multihop",
                chat_model=chat_model,
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

        return [[category(row[0]), row[0], row[1], row[2]] for row in rows]

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
                component_update(
                    value=(
                        '<div class="selected-document">'
                        f"<strong>{escape(record.relative_path)}</strong>"
                        f"<span>{record.page_count} page(s) · "
                        f"{record.chunk_count} chunk(s)</span></div>"
                    ),
                    visible=True,
                ),
                component_update(visible=True, interactive=True),
                "",
                component_update(visible=False),
            )
        else:
            selection = self.reset_document_selection(samples)
        return (
            component_update(samples=samples),
            self.corpus_summary_html(),
            component_update(label=f"Indexing errors ({len(error_rows)})"),
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
        return component_update(samples=samples), *self.reset_document_selection(samples)

    def index_selected_action_ui(
        self, files: list[str] | None, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        _documents, status, _errors, _readiness = self.index_selected(files, progress)
        return component_update(value=status, visible=True)

    def reindex_changed_action_ui(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        _documents, status, _errors = self.reindex_changed(progress)
        return component_update(value=status, visible=True)

    def rebuild_index_action_ui(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        _documents, status, _errors = self.rebuild_index(progress)
        return component_update(value=status, visible=True)

    def delete_selected_action_ui(self, document_id: str | None):
        _documents, status, _errors, text, confirmation = self.delete_selected(document_id)
        return component_update(value=status, visible=True), text, confirmation

    def index_selected_ui(
        self, files: list[str] | None, progress: gr.Progress = gr.Progress(track_tqdm=False)
    ):
        documents, status, errors, readiness = self.index_selected(files, progress)
        return (
            documents,
            component_update(value=status, visible=True),
            component_update(value=self.indexing_errors_html(errors), visible=bool(errors)),
            readiness,
        )

    def reindex_changed_ui(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        documents, status, errors = self.reindex_changed(progress)
        return (
            documents,
            component_update(value=status, visible=True),
            component_update(value=self.indexing_errors_html(errors), visible=bool(errors)),
        )

    def rebuild_index_ui(self, progress: gr.Progress = gr.Progress(track_tqdm=False)):
        documents, status, errors = self.rebuild_index(progress)
        return (
            documents,
            component_update(value=status, visible=True),
            component_update(value=self.indexing_errors_html(errors), visible=bool(errors)),
        )

    def reconcile_manifest_index_ui(self):
        documents, status, errors = self.reconcile_manifest_index()
        return (
            documents,
            component_update(value=status, visible=True),
            component_update(value=self.indexing_errors_html(errors), visible=bool(errors)),
        )

    def delete_selected_ui(self, document_id: str | None):
        documents, status, errors, text, confirmation = self.delete_selected(document_id)
        return (
            documents,
            component_update(value=status, visible=True),
            component_update(value=self.indexing_errors_html(errors), visible=bool(errors)),
            text,
            confirmation,
        )

    def chat_ui(self, message: str, history: list[dict[str, str]], session_id: str):
        values = list(self.chat(message, history, session_id))
        values[3] = component_update(value=values[3], visible=True)
        values[6] = self.scores_html(normalize_result_rows(values[6]))
        values[7] = self.trace_html(normalize_result_rows(values[7]))
        return tuple(values)

    def clear_ui(self, session_id: str):
        if self.rag_graph is not None:
            self.rag_graph.clear(session_id)
        return (
            [],
            {},
            component_update(value="", visible=False),
            "Conversation cleared.",
            self.evidence_html({}),
            self.scores_html([]),
            self.trace_html([]),
            str(uuid4()),
        )

    def export_chat_ui(self, messages: list[Any], result: dict[str, Any]):
        return component_update(
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

    def initialize_evaluation_ui(self, split: str, systems: list[str] | str | None):
        return self.load_latest_evaluation_ui(split, systems)

    def evaluation_options_ui(self, split: str, systems: list[str] | str | None):
        readiness = self.evaluation_readiness(split, systems)
        has_result = readiness.latest_result is not None
        if readiness.state == "blocked":
            status = component_update(
                value=render_status("warning", "Benchmark unavailable", readiness.problems[0]),
                visible=True,
            )
        else:
            status = component_update(value="", visible=False)
        return status, component_update(
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
            component_update(interactive=False),
            component_update(interactive=False),
            component_update(
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
            component_update(visible=has_result),
            component_update(value=context, visible=has_result),
            component_update(value=self.metrics_html(metrics), visible=has_result),
            self.failures_html(failures),
            component_update(visible=has_result),
            component_update(value=status, visible=True),
            component_update(
                value="Run new evaluation" if has_result else "Run standard benchmark",
                interactive=not blocked,
            ),
            component_update(interactive=True),
        )

    def preflight_presentation_ui(self):
        summary, rows, message_update, send_update = self.preflight_ui()
        return summary, self.system_status_html(rows), message_update, send_update

    def load_ai_models_presentation_ui(self):
        summary, rows, message_update, send_update = self.load_ai_models_ui()
        return summary, self.system_status_html(rows), message_update, send_update

    def preflight_shell_ui(self):
        summary, system_status, message_update, send_update = self.preflight_presentation_ui()
        return (
            summary,
            system_status,
            message_update,
            send_update,
            component_update(visible=self.rag_graph is None),
        )

    def load_ai_models_shell_ui(self):
        summary, system_status, message_update, send_update = self.load_ai_models_presentation_ui()
        return (
            summary,
            system_status,
            message_update,
            send_update,
            component_update(visible=self.rag_graph is None),
        )
