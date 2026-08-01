"""Production seven-system benchmark execution and presentation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Protocol, cast
from uuid import UUID

from langchain_ollama import ChatOllama
from pydantic import JsonValue

from modules.application.benchmark_manager import (
    BenchmarkCancellation,
    BenchmarkExecutionResult,
    BenchmarkReporter,
)
from modules.application.models import (
    BenchmarkCaseDetail,
    BenchmarkCaseMetricObservation,
    BenchmarkEventType,
    BenchmarkFailure,
    BenchmarkGraphConfiguration,
    BenchmarkMetadata,
    BenchmarkMetric,
    BenchmarkMetricObservation,
    BenchmarkProgress,
    BenchmarkReproducibility,
    BenchmarkSection,
    BenchmarkSystem,
    TraceEvent,
)
from modules.config import Settings, config
from modules.evaluation import (
    CountingModel,
    _require_ollama,
    dataset_content_hash,
    filter_cases,
    load_cases,
    map_retrieved_evidence,
    multihop_settings,
    preflight_multihop,
    required_models_for_systems,
    run_agentic_case,
    run_fixed_rag_case,
    run_retrieval_case,
)
from modules.evaluation_metrics import aggregate_metrics, failure_labels
from modules.evaluation_models import (
    ANSWER_SYSTEMS,
    CANONICAL_BENCHMARK_CASE_IDS,
    CANONICAL_BENCHMARK_RESULT_COUNT,
    CANONICAL_CHAT_MODEL,
    CANONICAL_CHUNK_OVERLAP,
    CANONICAL_CHUNK_SIZE,
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_MAX_CONTEXT_CHUNKS,
    CANONICAL_REQUEST_TIMEOUT_SECONDS,
    CANONICAL_RETRIEVAL_LIMIT,
    CANONICAL_RETRY_LIMIT,
    CANONICAL_SEMANTIC_CANDIDATES,
    CANONICAL_SPARSE_CANDIDATES,
    CANONICAL_SUBQUERY_LIMIT,
    CANONICAL_TEMPERATURE,
    FIXED_RAG_PROMPT_ID,
    FIXED_RAG_SYSTEMS,
    FULL_RAG_SYSTEM,
    MULTIHOP_ROOT,
    SYSTEMS,
    CaseResult,
    EvaluationCase,
    MetricObservation,
    SystemName,
)
from modules.evaluation_reporting import _git_commit
from modules.rag_graph import RAGGraph
from modules.retrieval import Retriever
from modules.vector_db import VectorDBManager

_SYSTEM_LABELS = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "dense-rag": "Dense RAG",
    "bm25-rag": "BM25 RAG",
    "hybrid-rag": "Hybrid RAG",
    "full-rag": "Full RAG",
}
_METRIC_LABELS = {
    "recall_at_5": "Recall at 5",
    "document_recall_at_5": "Document recall at 5",
    "mrr_at_5": "MRR at 5",
    "ndcg_at_5": "nDCG at 5",
    "citation_precision": "Citation precision",
    "gold_evidence_citation_coverage": "Evidence coverage",
    "abstention_accuracy": "Abstention accuracy",
    "unanswerable_abstention_recall": "Unanswerable abstention recall",
    "answerable_response_rate": "Answerable response rate",
    "normalized_answer_exact_match": "Normalized exact match",
    "answer_token_f1": "Answer token F1",
    "termination_rate": "Termination rate",
    "mean_latency_seconds": "Mean latency",
    "p95_latency_seconds": "P95 latency",
    "mean_llm_calls_per_query": "Mean LLM calls",
    "mean_retrieval_rounds_per_query": "Mean retrieval rounds",
    "route_accuracy": "Route accuracy",
    "strategy_accuracy": "Strategy accuracy",
    "retry_precision": "Retry precision",
    "retry_recall": "Retry recall",
    "conflict_recall": "Conflict recall",
    "conflict_false_positive_rate": "Conflict false-positive rate",
    "runtime_error_count": "Runtime error count",
    "runtime_error_rate": "Runtime error rate",
    "retrieval_miss_count": "Retrieval miss count",
    "citation_failure_count": "Citation failure count",
    "over_abstention_count": "Over-abstention count",
    "failed_abstention_count": "Failed abstention count",
    "non_termination_count": "Non-termination count",
    "route_failure_count": "Route failure count",
    "strategy_failure_count": "Strategy failure count",
    "retry_failure_count": "Retry failure count",
    "conflict_failure_count": "Conflict failure count",
}
_SECTION_METRICS = {
    "retrieval": (
        "recall_at_5",
        "document_recall_at_5",
        "mrr_at_5",
        "ndcg_at_5",
    ),
    "grounding": (
        "citation_precision",
        "gold_evidence_citation_coverage",
        "abstention_accuracy",
        "unanswerable_abstention_recall",
        "answerable_response_rate",
        "normalized_answer_exact_match",
        "answer_token_f1",
    ),
    "execution": (
        "termination_rate",
        "mean_latency_seconds",
        "p95_latency_seconds",
        "mean_llm_calls_per_query",
        "mean_retrieval_rounds_per_query",
        "route_accuracy",
        "strategy_accuracy",
        "retry_precision",
        "retry_recall",
        "conflict_recall",
        "conflict_false_positive_rate",
        "runtime_error_count",
        "runtime_error_rate",
        "retrieval_miss_count",
        "citation_failure_count",
        "over_abstention_count",
        "failed_abstention_count",
        "non_termination_count",
        "route_failure_count",
        "strategy_failure_count",
        "retry_failure_count",
        "conflict_failure_count",
    ),
}


class EvaluationRuntime(Protocol):
    def dataset_cases(self) -> list[EvaluationCase]: ...

    def prepare(self, chat_model: str) -> list[EvaluationCase]: ...

    def run_case(self, *, case: EvaluationCase, system: SystemName) -> CaseResult: ...


class LocalEvaluationRuntime:
    """Stateful synchronous adapter around the existing evaluator."""

    def __init__(self, *, settings: Settings = config) -> None:
        self.settings = settings
        self._retriever: Retriever | None = None
        self._model: CountingModel | None = None
        self._graph: RAGGraph | None = None

    def dataset_cases(self) -> list[EvaluationCase]:
        return load_cases(MULTIHOP_ROOT / "cases.jsonl")

    def prepare(self, chat_model: str) -> list[EvaluationCase]:
        manager = VectorDBManager(multihop_settings())
        cases = preflight_multihop(
            load_cases(MULTIHOP_ROOT / "cases.jsonl"),
            manager=manager,
            check_models=False,
        )
        development = filter_cases(cases, "development")
        _require_ollama(required_models_for_systems(SYSTEMS, chat_model))
        self._retriever = Retriever(manager.setup())
        self._model = CountingModel(
            ChatOllama(
                model=chat_model,
                base_url=self.settings.ollama_base_url,
                temperature=self.settings.temperature,
                num_predict=512,
                client_kwargs={"timeout": CANONICAL_REQUEST_TIMEOUT_SECONDS},
            )
        )
        self._graph = RAGGraph(manager, llm=self._model)  # type: ignore[arg-type]
        return development

    def run_case(self, *, case: EvaluationCase, system: SystemName) -> CaseResult:
        if self._retriever is None or self._model is None:
            raise RuntimeError("benchmark runtime has not been prepared")
        if system == FULL_RAG_SYSTEM:
            if self._graph is None:
                raise RuntimeError("full RAG graph has not been prepared")
            return run_agentic_case(case, self._graph, self._model)
        if system in FIXED_RAG_SYSTEMS:
            return run_fixed_rag_case(case, system, self._retriever, self._model)
        return run_retrieval_case(case, system, self._retriever)

    def set_cancellation_check(self, check: Callable[[], bool]) -> None:
        if self._model is not None:
            self._model.set_cancellation_check(check)


def _metric(
    *,
    name: str,
    observations: dict[SystemName, dict[str, MetricObservation]],
    systems: Sequence[SystemName],
) -> BenchmarkMetric:
    return BenchmarkMetric(
        name=name,
        label=_METRIC_LABELS.get(name, name.replace("_", " ").title()),
        observations=[
            BenchmarkMetricObservation(
                system=system,
                **observations[system][name].model_dump(),
            )
            for system in systems
        ],
    )


def _sections(
    observations: dict[SystemName, dict[str, MetricObservation]],
) -> list[BenchmarkSection]:
    return [
        BenchmarkSection(
            id="retrieval",
            title="Retrieval",
            system_ids=list(SYSTEMS),
            metrics=[
                _metric(name=name, observations=observations, systems=SYSTEMS)
                for name in _SECTION_METRICS["retrieval"]
            ],
            detail="Retrieval quality at five for every system's retrieval stage.",
        ),
        BenchmarkSection(
            id="grounding",
            title="Grounding",
            system_ids=list(ANSWER_SYSTEMS),
            metrics=[
                _metric(name=name, observations=observations, systems=ANSWER_SYSTEMS)
                for name in _SECTION_METRICS["grounding"]
            ],
            detail="Answer quality and evidence grounding for answer-producing systems.",
        ),
        BenchmarkSection(
            id="execution",
            title="Execution",
            system_ids=list(SYSTEMS),
            metrics=[
                _metric(name=name, observations=observations, systems=SYSTEMS)
                for name in _SECTION_METRICS["execution"]
            ],
            detail="Runtime characteristics and full-RAG workflow decisions.",
        ),
    ]


def _trace(result: CaseResult) -> list[TraceEvent]:
    if not result.public_trace:
        return [
            TraceEvent(
                stage="evaluate",
                decision="completed" if result.terminated else "failed",
                retrieved_count=len(result.retrieved_chunk_ids),
                fused_count=None,
                selected_count=len(result.retrieved_chunk_ids),
                retry_count=result.retry_count,
                llm_calls=result.llm_calls,
                termination=result.termination_reason
                or ("complete" if result.terminated else "failed"),
                duration_ms=result.latency_seconds * 1000,
            )
        ]
    return [
        TraceEvent(
            stage=str(event.get("stage") or "evaluate"),
            decision=str(event.get("decision") or "completed"),
            retrieved_count=cast(int | None, event.get("retrieved_count")),
            fused_count=cast(int | None, event.get("fused_count")),
            selected_count=cast(int | None, event.get("selected_count")),
            retry_count=cast(int, event.get("retry_count") or 0),
            llm_calls=cast(int, event.get("llm_calls") or 0),
            termination=str(event.get("termination") or "continue"),
            duration_ms=cast(float | None, event.get("duration_ms")),
        )
        for event in result.public_trace
    ]


def _case_detail(*, case: EvaluationCase, result: CaseResult) -> BenchmarkCaseDetail:
    metrics = aggregate_metrics([case], [result], system=result.system)
    expected_evidence = (
        [item.model_dump(mode="json") for item in case.gold_evidence]
        if case.gold_evidence
        else [
            {"chunk_id": chunk_id, "document_ids": list(case.relevant_document_ids)}
            for chunk_id in case.relevant_chunk_ids
        ]
    )
    classification = ", ".join(result.failure_labels) or None
    return BenchmarkCaseDetail(
        case_id=case.id,
        system=result.system,
        question=case.question,
        expected_answer=case.expected_answer,
        generated_answer=result.answer or None,
        expected_evidence=expected_evidence,
        retrieved_evidence=map_retrieved_evidence(
            result.retrieved_evidence
            or [{"chunk_id": chunk_id} for chunk_id in result.retrieved_chunk_ids]
        ),
        metric_observations=[
            BenchmarkCaseMetricObservation(
                name=name,
                label=_METRIC_LABELS.get(name, name.replace("_", " ").title()),
                system=result.system,
                **observation.model_dump(),
            )
            for name, observation in metrics.items()
        ],
        failure_classification=classification,
        public_trace=_trace(result),
        sanitized_raw_result={
            "failure_labels": list(result.failure_labels),
            "retrieval_rounds": result.retrieval_rounds,
            "llm_calls": result.llm_calls,
            "terminated": result.terminated,
            "abstained": result.abstained,
        },
    )


class FullRagBenchmarkExecutor:
    def __init__(
        self,
        *,
        runtime: EvaluationRuntime | None = None,
        chat_model_provider: Callable[[], str] | None = None,
        embedding_model: str | None = None,
        settings: Settings = config,
    ) -> None:
        self._runtime = runtime or LocalEvaluationRuntime(settings=settings)
        self._chat_model_provider = chat_model_provider or (lambda: settings.llm_model)
        self._embedding_model = embedding_model or settings.embedding_model
        self._settings = settings

    def _canonical_reproducibility(self) -> dict[str, JsonValue]:
        graph_configuration = BenchmarkGraphConfiguration(
            max_candidates=self._settings.max_candidates,
            maximum_context_chunks=self._settings.max_context_chunks,
            retry_limit=self._settings.max_retries,
            subquery_limit=self._settings.max_subqueries,
        )
        canonical_case_ids = set(CANONICAL_BENCHMARK_CASE_IDS)
        dataset_cases = [
            case for case in self._runtime.dataset_cases() if case.id in canonical_case_ids
        ]
        values = BenchmarkReproducibility(
            benchmark_name="full_rag_benchmark",
            git_commit=_git_commit(),
            dataset_identifier="yixuantt/MultiHopRAG",
            dataset_hash=dataset_content_hash(dataset_cases),
            case_ids=list(CANONICAL_BENCHMARK_CASE_IDS),
            expected_result_count=CANONICAL_BENCHMARK_RESULT_COUNT,
            chat_model=self._chat_model_provider(),
            embedding_model=self._embedding_model,
            temperature=self._settings.temperature,
            fixed_rag_prompt_id=FIXED_RAG_PROMPT_ID,
            graph_configuration=graph_configuration,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            retrieval_limit=CANONICAL_RETRIEVAL_LIMIT,
            semantic_candidates=self._settings.semantic_candidates,
            sparse_candidates=self._settings.sparse_candidates,
            maximum_context_chunks=self._settings.max_context_chunks,
            retry_limit=self._settings.max_retries,
            subquery_limit=self._settings.max_subqueries,
            request_timeout_seconds=CANONICAL_REQUEST_TIMEOUT_SECONDS,
        ).model_dump(mode="json")
        expected = {
            "chat_model": CANONICAL_CHAT_MODEL,
            "embedding_model": CANONICAL_EMBEDDING_MODEL,
            "temperature": CANONICAL_TEMPERATURE,
            "chunk_size": CANONICAL_CHUNK_SIZE,
            "chunk_overlap": CANONICAL_CHUNK_OVERLAP,
            "retrieval_limit": CANONICAL_RETRIEVAL_LIMIT,
            "semantic_candidates": CANONICAL_SEMANTIC_CANDIDATES,
            "sparse_candidates": CANONICAL_SPARSE_CANDIDATES,
            "maximum_context_chunks": CANONICAL_MAX_CONTEXT_CHUNKS,
            "retry_limit": CANONICAL_RETRY_LIMIT,
            "subquery_limit": CANONICAL_SUBQUERY_LIMIT,
            "request_timeout_seconds": CANONICAL_REQUEST_TIMEOUT_SECONDS,
        }
        if graph_configuration.model_dump(mode="json") != {
            "max_candidates": 20,
            "maximum_context_chunks": CANONICAL_MAX_CONTEXT_CHUNKS,
            "retry_limit": CANONICAL_RETRY_LIMIT,
            "subquery_limit": CANONICAL_SUBQUERY_LIMIT,
        }:
            raise ValueError(
                "The Full RAG Benchmark requires the canonical benchmark configuration."
            )
        if any(values[name] != expected_value for name, expected_value in expected.items()):
            raise ValueError(
                "The Full RAG Benchmark requires the canonical benchmark configuration."
            )
        return values

    def initial_metadata(self) -> BenchmarkMetadata:
        reproducibility = self._canonical_reproducibility()
        return BenchmarkMetadata(
            dataset="MultiHopRAG",
            split="development",
            systems=[
                BenchmarkSystem(id=system, label=_SYSTEM_LABELS[system]) for system in SYSTEMS
            ],
            chat_model=self._chat_model_provider(),
            embedding_model=self._embedding_model,
            started_at=None,
            completed_at=None,
            reproducibility=reproducibility,
        )

    async def _run_and_publish_case(
        self,
        *,
        case: EvaluationCase,
        system: SystemName,
        reporter: BenchmarkReporter,
        progress: BenchmarkProgress,
    ) -> tuple[CaseResult, BenchmarkFailure | None]:
        try:
            result = await asyncio.to_thread(
                self._runtime.run_case,
                case=case,
                system=system,
            )
        except Exception as exc:  # noqa: BLE001 - continue after isolated case failures
            result = CaseResult(
                case_id=case.id,
                system=system,
                runtime_error=type(exc).__name__,
            )
            result.failure_labels = failure_labels(case, result)
        failure = (
            BenchmarkFailure(
                case_id=case.id,
                system=system,
                classification=", ".join(result.failure_labels),
                detail=(
                    "Case execution failed."
                    if result.runtime_error
                    else "The case did not meet one or more benchmark expectations."
                ),
            )
            if result.failure_labels
            else None
        )
        await reporter.publish(
            (
                BenchmarkEventType.CASE_FAILED
                if result.runtime_error
                else BenchmarkEventType.CASE_COMPLETED
            ),
            {"case_id": case.id, "system": system},
            progress=progress,
            case=_case_detail(case=case, result=result),
        )
        return result, failure

    async def execute(  # lanorme: ignore[KWARG-001] -- BenchmarkExecutor protocol
        self,
        run_id: UUID,
        reporter: BenchmarkReporter,
        cancellation: BenchmarkCancellation,
    ) -> BenchmarkExecutionResult:
        del run_id
        self._canonical_reproducibility()
        chat_model = self._chat_model_provider()
        cases = await asyncio.to_thread(self._runtime.prepare, chat_model)
        set_cancellation_check = getattr(self._runtime, "set_cancellation_check", None)
        if set_cancellation_check is not None:
            set_cancellation_check(lambda: cancellation.is_cancelled)
        case_ids = [case.id for case in cases]
        if (
            len(case_ids) != len(CANONICAL_BENCHMARK_CASE_IDS)
            or set(case_ids) != set(CANONICAL_BENCHMARK_CASE_IDS)
            or any(case.split != "development" for case in cases)
        ):
            raise ValueError(
                "The Full RAG Benchmark requires the canonical 20-case development dataset."
            )
        total_cases = len(cases)
        results: list[CaseResult] = []
        failures: list[BenchmarkFailure] = []

        for system_index, system in enumerate(SYSTEMS, 1):
            if cancellation.is_cancelled:
                break
            progress = BenchmarkProgress(
                completed_cases=0,
                total_cases=total_cases,
                current_system=system,
                current_system_index=system_index,
                total_systems=len(SYSTEMS),
            )
            await reporter.publish(
                BenchmarkEventType.SYSTEM_STARTED,
                {"system": system},
                progress=progress,
            )
            for case_index, case in enumerate(cases, 1):
                if cancellation.is_cancelled:
                    break
                progress = progress.model_copy(
                    update={
                        "completed_cases": case_index - 1,
                        "current_case_id": case.id,
                        "current_case_index": case_index,
                    }
                )
                await reporter.publish(
                    BenchmarkEventType.CASE_STARTED,
                    {"case_id": case.id, "system": system},
                    progress=progress,
                )
                if cancellation.is_cancelled:
                    break
                result, failure = await self._run_and_publish_case(
                    case=case,
                    system=system,
                    reporter=reporter,
                    progress=progress.model_copy(update={"completed_cases": case_index}),
                )
                results.append(result)
                if failure is not None:
                    failures.append(failure)
            if cancellation.is_cancelled:
                break
            await reporter.publish(
                BenchmarkEventType.SYSTEM_COMPLETED,
                {"system": system},
                progress=progress.model_copy(
                    update={
                        "completed_cases": (
                            progress.current_case_index
                            if progress.current_case_index is not None
                            else 0
                        ),
                        "current_case_id": None,
                        "current_case_index": None,
                    }
                ),
            )

        if cancellation.is_cancelled:
            return BenchmarkExecutionResult(sections=[], failures=failures)

        observations: dict[SystemName, dict[str, MetricObservation]] = {}
        for system in SYSTEMS:
            observations[system] = aggregate_metrics(
                cases,
                [result for result in results if result.system == system],
                system=system,
            )
        return BenchmarkExecutionResult(
            sections=_sections(observations),
            failures=failures,
        )
