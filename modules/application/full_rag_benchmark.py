"""Production seven-system benchmark execution and presentation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from langchain_ollama import ChatOllama
from pydantic import JsonValue

from modules.application.benchmark_manager import (
    BenchmarkCancellation,
    BenchmarkExecutionResult,
    BenchmarkReporter,
)
from modules.application.benchmark_presentation import benchmark_case_detail, benchmark_sections
from modules.application.benchmark_reproducibility import (
    BenchmarkRuntimeIdentity,
    canonical_reproducibility,
)
from modules.application.models import (
    BenchmarkEventType,
    BenchmarkFailure,
    BenchmarkMetadata,
    BenchmarkProgress,
    BenchmarkSystem,
)
from modules.config import Settings, config
from modules.evaluation import (
    CountingModel,
    _require_ollama,
    filter_cases,
    load_cases,
    multihop_settings,
    preflight_multihop,
    required_models_for_systems,
    run_agentic_case,
    run_fixed_rag_case,
    run_retrieval_case,
)
from modules.evaluation_metrics import aggregate_metrics, failure_labels
from modules.evaluation_models import (
    CANONICAL_BENCHMARK_CASE_IDS,
    CANONICAL_REQUEST_TIMEOUT_SECONDS,
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


@dataclass(frozen=True)
class _ExecutionContext:
    cases: list[EvaluationCase]
    reporter: BenchmarkReporter
    cancellation: BenchmarkCancellation


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
        canonical_case_ids = set(CANONICAL_BENCHMARK_CASE_IDS)
        dataset_cases = [
            case for case in self._runtime.dataset_cases() if case.id in canonical_case_ids
        ]
        return canonical_reproducibility(
            settings=self._settings,
            cases=dataset_cases,
            identity=BenchmarkRuntimeIdentity(
                git_commit=_git_commit(),
                chat_model=self._chat_model_provider(),
                embedding_model=self._embedding_model,
            ),
        )

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
            case=benchmark_case_detail(case=case, result=result),
        )
        return result, failure

    @staticmethod
    def _validate_cases(cases: list[EvaluationCase]) -> None:
        case_ids = [case.id for case in cases]
        if (
            len(case_ids) != len(CANONICAL_BENCHMARK_CASE_IDS)
            or set(case_ids) != set(CANONICAL_BENCHMARK_CASE_IDS)
            or any(case.split != "development" for case in cases)
        ):
            raise ValueError(
                "The Full RAG Benchmark requires the canonical 20-case development dataset."
            )

    @staticmethod
    async def _start_system(*, system: SystemName, context: _ExecutionContext) -> BenchmarkProgress:
        progress = BenchmarkProgress(
            completed_cases=0,
            total_cases=len(context.cases),
            current_system=system,
            current_system_index=SYSTEMS.index(system) + 1,
            total_systems=len(SYSTEMS),
        )
        await context.reporter.publish(
            BenchmarkEventType.SYSTEM_STARTED,
            {"system": system},
            progress=progress,
        )
        return progress

    @staticmethod
    async def _complete_system(
        *, system: SystemName, reporter: BenchmarkReporter, progress: BenchmarkProgress
    ) -> None:
        await reporter.publish(
            BenchmarkEventType.SYSTEM_COMPLETED,
            {"system": system},
            progress=progress.model_copy(
                update={
                    "completed_cases": progress.current_case_index or 0,
                    "current_case_id": None,
                    "current_case_index": None,
                }
            ),
        )

    async def _run_system(
        self,
        *,
        system: SystemName,
        context: _ExecutionContext,
    ) -> tuple[list[CaseResult], list[BenchmarkFailure]]:
        cases = context.cases
        reporter = context.reporter
        cancellation = context.cancellation
        progress = await self._start_system(system=system, context=context)
        results: list[CaseResult] = []
        failures: list[BenchmarkFailure] = []
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
        if not cancellation.is_cancelled:
            await self._complete_system(system=system, reporter=reporter, progress=progress)
        return results, failures

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
        self._validate_cases(cases)
        results: list[CaseResult] = []
        failures: list[BenchmarkFailure] = []
        context = _ExecutionContext(
            cases=cases,
            reporter=reporter,
            cancellation=cancellation,
        )

        for system in SYSTEMS:
            if cancellation.is_cancelled:
                break
            system_results, system_failures = await self._run_system(
                system=system,
                context=context,
            )
            results.extend(system_results)
            failures.extend(system_failures)

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
            sections=benchmark_sections(observations),
            failures=failures,
        )
