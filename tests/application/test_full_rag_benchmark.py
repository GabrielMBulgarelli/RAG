import asyncio
from uuid import uuid4

from pydantic import JsonValue

from modules.application.benchmark_manager import BenchmarkCancellation
from modules.application.full_rag_benchmark import FullRagBenchmarkExecutor
from modules.application.models import (
    BenchmarkCaseDetail,
    BenchmarkEventType,
    BenchmarkProgress,
)
from modules.evaluation_models import SYSTEMS, CaseResult, EvaluationCase, SystemName


def benchmark_case(case_id: str = "case-1") -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        split="development",
        category="comparison",
        question="Which organization changed its policy?",
        answerable=True,
        relevant_chunk_ids=["chunk-1"],
        relevant_document_ids=["doc-1"],
        expected_answer="Example Org",
        expected_route="complex_search",
        expected_strategy="hybrid",
        expected_retry=False,
        expected_conflict=False,
    )


class FakeRuntime:
    def __init__(self, cases: list[EvaluationCase]) -> None:
        self.cases = cases
        self.calls: list[tuple[str, SystemName]] = []

    def prepare(self, chat_model: str) -> list[EvaluationCase]:
        del chat_model
        return self.cases

    def run_case(self, *, case: EvaluationCase, system: SystemName) -> CaseResult:
        self.calls.append((case.id, system))
        return CaseResult(
            case_id=case.id,
            system=system,
            retrieved_chunk_ids=["chunk-1"],
            retrieved_document_ids=["doc-1"],
            cited_chunk_ids=["chunk-1"] if system.endswith("rag") else [],
            route="complex_search" if system == "full-rag" else None,
            strategy="hybrid" if system == "full-rag" else None,
            terminated=True,
            answer="Example Org [C1]" if system.endswith("rag") else "",
            latency_seconds=0.1,
            llm_calls=1 if system.endswith("rag") else 0,
            retrieval_rounds=1,
            retrieved_evidence=[
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "filename": "document.pdf",
                    "page": 2,
                    "excerpt": "Example Org changed its policy.",
                }
            ],
        )


class RecordingReporter:
    def __init__(self, cancellation: BenchmarkCancellation | None = None) -> None:
        self.events: list[
            tuple[BenchmarkEventType, dict[str, JsonValue], BenchmarkCaseDetail | None]
        ] = []
        self.cancellation = cancellation

    async def publish(
        self,
        event_type: BenchmarkEventType,
        data: dict[str, JsonValue],
        *,
        progress: BenchmarkProgress,
        case: BenchmarkCaseDetail | None = None,
    ) -> None:
        del progress
        self.events.append((event_type, data, case))
        if self.cancellation is not None and event_type is BenchmarkEventType.CASE_COMPLETED:
            self.cancellation.request()


def test_executor_runs_exactly_seven_systems_and_builds_presenter_sections() -> None:
    # Arrange
    runtime = FakeRuntime([benchmark_case()])
    executor = FullRagBenchmarkExecutor(
        runtime=runtime,
        chat_model_provider=lambda: "qwen3:9b",
        embedding_model="nomic-embed-text",
    )
    reporter = RecordingReporter()

    # Act
    result = asyncio.run(executor.execute(uuid4(), reporter, BenchmarkCancellation()))

    # Then all seven systems and both benchmark families remain explicit
    assert [system.id for system in executor.initial_metadata().systems] == list(SYSTEMS)
    assert runtime.calls == [("case-1", system) for system in SYSTEMS]
    assert [section.id for section in result.sections] == [
        "retrieval",
        "grounding",
        "execution",
    ]
    assert result.sections[0].system_ids == list(SYSTEMS)
    assert result.sections[1].system_ids == [
        "dense-rag",
        "bm25-rag",
        "hybrid-rag",
        "full-rag",
    ]
    completed = [
        case
        for event_type, _data, case in reporter.events
        if event_type is BenchmarkEventType.CASE_COMPLETED and case is not None
    ]
    assert len(completed) == 7
    assert completed[-1].metric_observations[0].name
    assert completed[-1].metric_observations[0].label
    progress = reporter.events[-1][1]
    assert progress["system"] == "full-rag"


def test_executor_checks_cancellation_between_cases() -> None:
    # Arrange
    cancellation = BenchmarkCancellation()
    runtime = FakeRuntime([benchmark_case("case-1"), benchmark_case("case-2")])
    reporter = RecordingReporter(cancellation)
    executor = FullRagBenchmarkExecutor(
        runtime=runtime,
        chat_model_provider=lambda: "qwen3:9b",
        embedding_model="nomic-embed-text",
    )

    # Act
    asyncio.run(executor.execute(uuid4(), reporter, cancellation))

    # Then no second case or system starts
    assert runtime.calls == [("case-1", "dense")]


def test_executor_sanitizes_runtime_failures_and_continues() -> None:
    # Arrange
    class FailingRuntime(FakeRuntime):
        def run_case(self, *, case: EvaluationCase, system: SystemName) -> CaseResult:
            if system == "bm25":
                self.calls.append((case.id, system))
                raise RuntimeError("private path C:/Users/example")
            return super().run_case(case=case, system=system)

    runtime = FailingRuntime([benchmark_case()])
    reporter = RecordingReporter()
    executor = FullRagBenchmarkExecutor(
        runtime=runtime,
        chat_model_provider=lambda: "qwen3:9b",
        embedding_model="nomic-embed-text",
    )

    # Act
    result = asyncio.run(executor.execute(uuid4(), reporter, BenchmarkCancellation()))

    # Then only public failure data is retained and later systems still run
    assert len(runtime.calls) == 7
    assert result.failures[0].case_id == "case-1"
    assert result.failures[0].system == "bm25"
    assert result.failures[0].detail == "Case execution failed."
    failed_cases = [
        case
        for event_type, _data, case in reporter.events
        if event_type is BenchmarkEventType.CASE_FAILED and case is not None
    ]
    assert failed_cases[0].sanitized_raw_result == {
        "failure_labels": ["retrieval_miss", "non_termination", "runtime_error"],
        "retrieval_rounds": 0,
        "llm_calls": 0,
        "terminated": False,
        "abstained": False,
    }
