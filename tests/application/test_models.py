from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from modules.application.models import (
    ActiveOperation,
    ApiProblem,
    BenchmarkCaseDetail,
    BenchmarkEvent,
    BenchmarkEventType,
    BenchmarkFailure,
    BenchmarkMetadata,
    BenchmarkMetric,
    BenchmarkMetricObservation,
    BenchmarkMetricStatus,
    BenchmarkProgress,
    BenchmarkRun,
    BenchmarkRunStatus,
    BenchmarkSection,
    BenchmarkStartResponse,
    BenchmarkSystem,
    CapabilitySnapshot,
    ConversationExportRequest,
    ConversationMessage,
    CorpusSnapshot,
    DiagnosticCheck,
    DiagnosticsSnapshot,
    DocumentList,
    DocumentRecord,
    ModelLoadRequest,
    OperationKind,
    QueryDiagnostics,
    QueryRequest,
    QueryResponse,
    ResourceLinks,
    RetrievalHit,
    RuntimeSnapshot,
    Source,
    TraceEvent,
    UploadAccepted,
    UploadBatchResult,
)


def test_operation_models_have_exact_values_and_json_serialization() -> None:
    assert [kind.value for kind in OperationKind] == [
        "index_documents",
        "delete_document",
        "load_model",
        "query",
        "benchmark",
    ]

    operation = ActiveOperation(
        operation_id=UUID("0b5e8a4b-a5df-4a93-a796-4d8cce1f4367"),
        kind=OperationKind.BENCHMARK,
        started_at=datetime(2026, 7, 29, 15, 30, tzinfo=timezone.utc),
        benchmark_run_id=UUID("4cbdbcb9-5a57-4514-a392-2dce907456d5"),
    )

    assert operation.model_dump(mode="json") == {
        "operation_id": "0b5e8a4b-a5df-4a93-a796-4d8cce1f4367",
        "kind": "benchmark",
        "started_at": "2026-07-29T15:30:00Z",
        "benchmark_run_id": "4cbdbcb9-5a57-4514-a392-2dce907456d5",
        "cancellation_requested": False,
    }

    with pytest.raises(ValidationError, match="frozen"):
        operation.cancellation_requested = True


def test_runtime_document_and_request_contracts_serialize_as_json() -> None:
    operation = ActiveOperation(
        operation_id=UUID("0b5e8a4b-a5df-4a93-a796-4d8cce1f4367"),
        kind=OperationKind.INDEX_DOCUMENTS,
        started_at=datetime(2026, 7, 29, 15, 30, tzinfo=timezone.utc),
    )
    capabilities = CapabilitySnapshot(
        can_query=False,
        can_load_models=True,
        can_upload=True,
        can_run_benchmark=False,
    )
    corpus = CorpusSnapshot(
        document_count=1,
        page_count=2,
        chunk_count=3,
        status="ready",
    )
    runtime = RuntimeSnapshot(
        state="not_loaded",
        configured_chat_model="qwen3:4b",
        active_chat_model=None,
        embedding_model="nomic-embed-text",
        available_chat_models=["qwen3:4b"],
        detail="Load the configured chat model.",
        capabilities=capabilities,
        active_operation=operation,
        corpus=corpus,
    )
    check = DiagnosticCheck(
        area="runtime",
        name="Chat model",
        state="not_loaded",
        detail="No active chat model.",
    )
    diagnostics = DiagnosticsSnapshot(
        state="blocked",
        title="Workspace needs attention",
        detail="Load the configured chat model.",
        runtime_checks=[check],
        index_checks=[],
        evaluation_checks=[],
        active_operation=operation,
        stale=False,
    )
    document = DocumentRecord(
        id=UUID("fbfabdb1-d47d-40d5-a9bd-76e20f383742"),
        filename="guide.pdf",
        state="indexed",
        size_bytes=4096,
        page_count=2,
        chunk_count=3,
        indexed_at=datetime(2026, 7, 29, 15, 31, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 29, 15, 31, tzinfo=timezone.utc),
    )
    documents = DocumentList(
        documents=[document],
        corpus=corpus,
        active_operation=operation,
    )
    upload = UploadBatchResult(
        accepted=[UploadAccepted(filename="guide.pdf", document_id=document.id)],
        documents=documents.documents,
        corpus=corpus,
    )

    assert runtime.model_dump(mode="json")["active_operation"]["kind"] == "index_documents"
    assert diagnostics.model_dump(mode="json")["runtime_checks"][0]["state"] == "not_loaded"
    assert documents.model_dump(mode="json")["documents"][0]["id"] == str(document.id)
    assert upload.model_dump(mode="json")["accepted"] == [
        {"filename": "guide.pdf", "document_id": str(document.id)}
    ]
    assert (
        ConversationMessage(
            id=UUID("b3c19351-5ae1-48e3-938f-c9306359478f"),
            role="assistant",
            content="Ready.",
            created_at=datetime(2026, 7, 29, 15, 32, tzinfo=timezone.utc),
        ).model_dump(mode="json")["role"]
        == "assistant"
    )
    assert ConversationExportRequest(
        session_id=UUID("26bc395b-c7ae-4111-a1e9-bb3677de67f7")
    ).model_dump(mode="json") == {"session_id": "26bc395b-c7ae-4111-a1e9-bb3677de67f7"}
    assert ModelLoadRequest(chat_model="  qwen3:4b  ").chat_model == "qwen3:4b"


def test_model_load_request_rejects_blank_chat_model() -> None:
    with pytest.raises(ValidationError, match="chat_model"):
        ModelLoadRequest(chat_model=" \t ")


def test_query_contract_serializes_public_observability() -> None:
    session_id = UUID("26bc395b-c7ae-4111-a1e9-bb3677de67f7")
    response = QueryResponse(
        session_id=session_id,
        message=ConversationMessage(
            id=UUID("b3c19351-5ae1-48e3-938f-c9306359478f"),
            role="assistant",
            content="The documented limit is 4.",
            created_at=datetime(2026, 7, 29, 15, 32, tzinfo=timezone.utc),
        ),
        answer_state="supported",
        sources=[
            Source(
                label="C1",
                filename="guide.pdf",
                page=2,
                excerpt="The limit is 4.",
            )
        ],
        retrieval_hits=[
            RetrievalHit(
                chunk_id="chunk-1",
                filename="guide.pdf",
                page=2,
                semantic_score=0.8,
                sparse_score=None,
                fused_score=0.7,
                selection_score=0.9,
                matched_subqueries=["documented limit"],
            )
        ],
        trace=[
            TraceEvent(
                stage="retrieve",
                decision="hybrid",
                retrieved_count=5,
                fused_count=4,
                selected_count=1,
                retry_count=0,
                llm_calls=0,
                termination="continue",
                duration_ms=12.5,
            )
        ],
        diagnostics=QueryDiagnostics(
            route="simple_search",
            retrieval_strategy="hybrid",
            subqueries=["documented limit"],
            retry_count=0,
            evidence_state="sufficient",
            conflict_state="none",
            citation_validation="valid",
        ),
    )

    assert QueryRequest(session_id=session_id, question="What is the limit?").model_dump(
        mode="json"
    ) == {
        "session_id": str(session_id),
        "question": "What is the limit?",
    }
    assert response.model_dump(mode="json")["retrieval_hits"][0] == {
        "chunk_id": "chunk-1",
        "filename": "guide.pdf",
        "page": 2,
        "semantic_score": 0.8,
        "sparse_score": None,
        "fused_score": 0.7,
        "selection_score": 0.9,
        "matched_subqueries": ["documented limit"],
    }
    assert ApiProblem(
        code="invalid_request",
        message="The request is invalid.",
        details={"field": "question"},
    ).model_dump(mode="json") == {
        "code": "invalid_request",
        "message": "The request is invalid.",
        "details": {"field": "question"},
    }


def test_benchmark_contract_has_exact_states_and_json_serialization() -> None:
    assert [status.value for status in BenchmarkRunStatus] == [
        "queued",
        "running",
        "cancellation_requested",
        "cancelled",
        "completed",
        "failed",
    ]
    assert [event_type.value for event_type in BenchmarkEventType] == [
        "benchmark.started",
        "system.started",
        "case.started",
        "case.completed",
        "case.failed",
        "system.completed",
        "benchmark.cancellation_requested",
        "benchmark.cancelled",
        "benchmark.completed",
        "benchmark.failed",
        "heartbeat",
    ]

    run_id = UUID("4cbdbcb9-5a57-4514-a392-2dce907456d5")
    links = ResourceLinks(
        run=f"/api/benchmarks/{run_id}",
        events=f"/api/benchmarks/{run_id}/events",
        download=f"/api/benchmarks/{run_id}/download",
    )
    observation = BenchmarkMetricObservation(
        system="dense",
        value=0.75,
        status=BenchmarkMetricStatus.MEASURED,
        sample_count=4,
        note=None,
    )
    benchmark = BenchmarkRun(
        run_id=run_id,
        status=BenchmarkRunStatus.RUNNING,
        progress=BenchmarkProgress(
            completed_cases=1,
            total_cases=4,
            current_system="dense",
            current_case_id="case-2",
        ),
        metadata=BenchmarkMetadata(
            dataset="multihop",
            split="development",
            systems=[BenchmarkSystem(id="dense", label="Dense")],
            chat_model="qwen3:4b",
            embedding_model="nomic-embed-text",
            started_at=datetime(2026, 7, 29, 15, 30, tzinfo=timezone.utc),
            completed_at=None,
        ),
        sections=[
            BenchmarkSection(
                id="retrieval",
                title="Retrieval",
                metrics=[
                    BenchmarkMetric(
                        name="recall_at_5",
                        label="Recall at 5",
                        observations=[observation],
                    )
                ],
            )
        ],
        failures=[
            BenchmarkFailure(
                case_id="case-1",
                system="dense",
                classification="retrieval_miss",
                detail="Expected evidence was not retrieved.",
            )
        ],
        links=links,
        error=None,
    )
    event = BenchmarkEvent(
        event_id=UUID("ce81a5d9-5b28-4589-b5ec-792493538590"),
        run_id=run_id,
        type=BenchmarkEventType.CASE_COMPLETED,
        timestamp=datetime(2026, 7, 29, 15, 31, tzinfo=timezone.utc),
        data={"case_id": "case-1", "score": 0.75},
    )
    case = BenchmarkCaseDetail(
        case_id="case-1",
        system="dense",
        question="What is the limit?",
        expected_answer="4",
        generated_answer="The limit is 4.",
        expected_evidence=[{"document_id": "guide", "text": "The limit is 4."}],
        retrieved_evidence=[{"chunk_id": "chunk-1", "text": "The limit is 4."}],
        metric_observations=[observation],
        failure_classification=None,
        public_trace=[],
        sanitized_raw_result={"route": "simple_search"},
    )

    assert (
        BenchmarkStartResponse(
            run_id=run_id,
            status=BenchmarkRunStatus.QUEUED,
            links=links,
        ).model_dump(mode="json")["status"]
        == "queued"
    )
    assert (
        benchmark.model_dump(mode="json")["sections"][0]["metrics"][0]["observations"][0]["value"]
        == 0.75
    )
    assert event.model_dump(mode="json")["type"] == "case.completed"
    assert case.model_dump(mode="json")["sanitized_raw_result"] == {"route": "simple_search"}


def test_metric_status_keeps_missing_values_distinct_from_measured_zero() -> None:
    missing = BenchmarkMetricObservation(
        system="dense",
        value=None,
        status=BenchmarkMetricStatus.NO_ELIGIBLE_CASES,
        sample_count=0,
    )
    measured_zero = BenchmarkMetricObservation(
        system="dense",
        value=0.0,
        status=BenchmarkMetricStatus.MEASURED,
        sample_count=1,
    )

    assert missing.model_dump(mode="json")["value"] is None
    assert measured_zero.model_dump(mode="json")["value"] == 0.0
    with pytest.raises(ValidationError, match="measured metrics require a value"):
        BenchmarkMetricObservation(
            system="dense",
            value=None,
            status=BenchmarkMetricStatus.MEASURED,
            sample_count=1,
        )
    with pytest.raises(ValidationError, match="unmeasured metrics must use a null value"):
        BenchmarkMetricObservation(
            system="dense",
            value=0.0,
            status=BenchmarkMetricStatus.NOT_APPLICABLE,
            sample_count=0,
        )
