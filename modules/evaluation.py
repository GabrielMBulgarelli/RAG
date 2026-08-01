"""Comparative evaluation harness for retrieval, fixed RAG, and full RAG systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from pydantic import JsonValue

from modules.citations import build_cited_context, retain_cited_claims, validate_answer
from modules.config import PROJECT_ROOT, Settings, config
from modules.evaluation_metrics import (
    aggregate_metrics,
    citation_precision,
    failure_labels,
    filter_cases,
    gold_citation_coverage,
    mrr_at_k,
    ndcg_at_k,
    normalized_exact_match,
    p95,
    recall_at_k,
    token_f1,
)
from modules.evaluation_models import (
    ANSWER_SYSTEMS,
    CANONICAL_REQUEST_TIMEOUT_SECONDS,
    FIXED_RAG_PROMPT_ID,
    FIXED_RAG_SYSTEMS,
    FULL_RAG_SYSTEM,
    MULTIHOP_ROOT,
    STANDARD_BENCHMARK_DATASET,
    STANDARD_BENCHMARK_SPLIT,
    SYSTEMS,
    BenchmarkEvidence,
    CaseResult,
    EvaluationCase,
    EvaluationResultKind,
    ExperimentConfig,
    MetricObservation,
    MetricStatus,
    Split,
    SystemName,
    evaluation_result_kind,
    is_complete_full_rag_benchmark_artifact,
)
from modules.evaluation_reporting import _git_commit, write_experiment
from modules.models import RetrievalHit
from modules.rag_graph import RAGGraph
from modules.retrieval import Retriever, reciprocal_rank_fusion
from modules.vector_db import VectorDBManager

__all__ = [
    "BenchmarkEvidence",
    "CaseResult",
    "CountingModel",
    "EvaluationCase",
    "EvaluationResultKind",
    "ExperimentConfig",
    "MetricObservation",
    "MetricStatus",
    "STANDARD_BENCHMARK_DATASET",
    "STANDARD_BENCHMARK_SPLIT",
    "SYSTEMS",
    "Split",
    "SystemName",
    "aggregate_metrics",
    "citation_precision",
    "evaluation_result_kind",
    "failure_labels",
    "filter_cases",
    "gold_citation_coverage",
    "is_complete_full_rag_benchmark_artifact",
    "map_retrieved_evidence",
    "mrr_at_k",
    "ndcg_at_k",
    "normalized_exact_match",
    "p95",
    "recall_at_k",
    "run_agentic_case",
    "run_evaluation",
    "run_fixed_rag_case",
    "run_retrieval_case",
    "token_f1",
    "write_experiment",
]


class _CallState:
    def __init__(self, cancellation_check: Callable[[], bool] | None = None) -> None:
        self.calls = 0
        self.cancellation_check = cancellation_check


class EvaluationCancelled(RuntimeError):
    """Raised after a cooperative benchmark cancellation is observed."""


class EvaluationTimeout(TimeoutError):
    """Raised when an agentic evaluation case exceeds its wall-clock budget."""


def _process_query_with_deadline(  # lanorme: ignore[KWARG-001,TYPE-001] -- Mirrors dynamic graph payload
    graph: RAGGraph, question: str, session_id: str, timeout_seconds: float
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("Case timeout must be greater than zero.")

    return graph.process_query(question, session_id)


class CountingModel:
    """Thin proxy counting all calls, including structured-output calls, at one boundary."""

    def __init__(
        self,
        model: Any,
        state: _CallState | None = None,
        *,
        cancellation_check: Callable[[], bool] | None = None,
    ):
        self._model = model
        self._state = state or _CallState(cancellation_check)

    def set_cancellation_check(self, check: Callable[[], bool]) -> None:
        self._state.cancellation_check = check

    @property
    def calls(self) -> int:
        return self._state.calls

    def invoke(self, value: object, **kwargs: object) -> Any:
        if self._state.cancellation_check and self._state.cancellation_check():
            raise EvaluationCancelled("benchmark cancellation requested")
        self._state.calls += 1
        result = self._model.invoke(value, **kwargs)
        if self._state.cancellation_check and self._state.cancellation_check():
            raise EvaluationCancelled("benchmark cancellation requested")
        return result

    def with_structured_output(self, schema: object, **kwargs: object) -> CountingModel:
        return CountingModel(self._model.with_structured_output(schema, **kwargs), self._state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def load_cases(path: Path) -> list[EvaluationCase]:
    return [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _retrieval_result(
    case: EvaluationCase, system: SystemName, hits: list[RetrievalHit], elapsed: float
) -> CaseResult:
    result = CaseResult(
        case_id=case.id,
        system=system,
        retrieved_chunk_ids=[hit.chunk_id for hit in hits[:5]],
        retrieved_document_ids=list(
            dict.fromkeys(hit.document_id for hit in hits[:5] if hit.document_id)
        ),
        terminated=True,
        abstained=not bool(hits),
        latency_seconds=elapsed,
        retrieval_rounds=1,
        retrieved_evidence=map_retrieved_evidence(hits),
    )
    result.failure_labels = failure_labels(case, result)
    return result


def run_retrieval_case(
    case: EvaluationCase, system: SystemName, retriever: Retriever
) -> CaseResult:
    started = time.perf_counter()
    if system == "dense":
        hits = retriever.semantic(case.question, 5)
    elif system == "bm25":
        hits = retriever.sparse(case.question, 5)
    elif system == "hybrid":
        dense = retriever.semantic(case.question, config.semantic_candidates)
        sparse = retriever.sparse(case.question, config.sparse_candidates)
        hits = reciprocal_rank_fusion(dense, sparse, limit=5)
    else:
        raise ValueError(f"Unsupported retrieval-only system: {system}")
    return _retrieval_result(case, system, hits, time.perf_counter() - started)


def _fixed_rag_hits(
    *, question: str, system: SystemName, retriever: Retriever
) -> list[RetrievalHit]:
    if system == "dense-rag":
        return retriever.semantic(question, 5)
    if system == "bm25-rag":
        return retriever.sparse(question, 5)
    if system == "hybrid-rag":
        dense = retriever.semantic(question, config.semantic_candidates)
        sparse = retriever.sparse(question, config.sparse_candidates)
        return reciprocal_rank_fusion(dense, sparse, limit=5)
    raise ValueError(f"Unsupported fixed RAG system: {system}")


def map_retrieved_evidence(
    hits: Sequence[RetrievalHit | Mapping[str, object]],
) -> list[dict[str, JsonValue]]:
    evidence: list[dict[str, JsonValue]] = []
    for raw_hit in hits[:5]:
        hit = raw_hit.model_dump(mode="json") if isinstance(raw_hit, RetrievalHit) else raw_hit
        text = hit.get("excerpt") or hit.get("content") or ""
        page = hit.get("page")
        evidence.append(
            {
                "chunk_id": str(hit.get("chunk_id") or ""),
                "document_id": str(hit.get("document_id") or ""),
                "filename": str(hit.get("filename") or ""),
                "page": page if isinstance(page, int) else None,
                "excerpt": " ".join(str(text).split())[:300],
            }
        )
    return evidence


def _message_text(value: object) -> str:
    content = getattr(value, "content", value)
    return content if isinstance(content, str) else str(content)


def run_fixed_rag_case(  # lanorme: ignore[KWARG-001] -- Stable evaluator API
    case: EvaluationCase,
    system: SystemName,
    retriever: Retriever,
    model: CountingModel,
) -> CaseResult:
    """Run fixed retrieval followed by at most one grounded answer call."""
    started_calls = model.calls
    started = time.perf_counter()
    try:
        hits = _fixed_rag_hits(question=case.question, system=system, retriever=retriever)
        context, sources = build_cited_context(hits)
        if not hits:
            answer = "I could not find enough evidence in the indexed documents to answer."
            validation = validate_answer(answer, [], known_labels=set(), require_citations=False)
            cited_sources = []
            abstained = True
        else:
            prompt = f"""Begin with a direct, concise answer. Use only the evidence below and
cite every factual claim with its exact [C#] label. Do not invent labels or facts.
Question: {case.question}
Evidence:
{context}"""
            raw_answer = _message_text(model.invoke([HumanMessage(content=prompt)], think=False))
            known_labels = {source.label for source in sources}
            validation = validate_answer(
                raw_answer,
                sources,
                known_labels=known_labels,
                require_citations=True,
            )
            grounded = validation
            if not validation.is_valid:
                grounded = retain_cited_claims(
                    raw_answer,
                    sources,
                    known_labels=known_labels,
                )
            if grounded.is_valid and grounded.used_sources:
                answer = grounded.sanitized_text
                cited_sources = grounded.used_sources
                abstained = False
            else:
                answer = "I could not produce a fully cited answer from the available evidence."
                cited_sources = []
                abstained = True
        result = CaseResult(
            case_id=case.id,
            system=system,
            retrieved_chunk_ids=[hit.chunk_id for hit in hits[:5]],
            retrieved_document_ids=list(
                dict.fromkeys(hit.document_id for hit in hits[:5] if hit.document_id)
            ),
            cited_chunk_ids=[source.chunk_id for source in cited_sources],
            terminated=True,
            abstained=abstained,
            latency_seconds=time.perf_counter() - started,
            llm_calls=model.calls - started_calls,
            retrieval_rounds=1,
            answer=answer,
            validation_violations=[item.value for item in validation.violations],
            retrieved_evidence=map_retrieved_evidence(hits),
        )
    except Exception as exc:  # noqa: BLE001 - evaluation must preserve per-case failures
        result = CaseResult(
            case_id=case.id,
            system=system,
            latency_seconds=time.perf_counter() - started,
            llm_calls=model.calls - started_calls,
            runtime_error=f"{type(exc).__name__}: {exc}",
        )
    result.failure_labels = failure_labels(case, result)
    return result


def run_agentic_case(  # lanorme: ignore[KWARG-001,SIZE-002] -- Stable evaluator API
    case: EvaluationCase,
    graph: RAGGraph,
    model: CountingModel,
    *,
    timeout_seconds: float = 30.0,
) -> CaseResult:
    started_calls = model.calls
    started = time.perf_counter()
    try:
        payload = _process_query_with_deadline(
            graph,
            case.question,
            f"evaluation-{case.id}",
            timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - evaluation must preserve per-case failures
        result = CaseResult(
            case_id=case.id,
            system=FULL_RAG_SYSTEM,
            latency_seconds=time.perf_counter() - started,
            llm_calls=model.calls - started_calls,
            runtime_error=f"{type(exc).__name__}: {exc}",
        )
        result.failure_labels = failure_labels(case, result)
        return result
    trace = payload.get("trace", [])
    termination = next(
        (
            event.get("termination")
            for event in reversed(trace)
            if event.get("stage") == "terminate"
        ),
        None,
    )
    validation = payload.get("validation", {})
    result = CaseResult(
        case_id=case.id,
        system=FULL_RAG_SYSTEM,
        retrieved_chunk_ids=[hit["chunk_id"] for hit in payload.get("retrieval_hits", [])][:5],
        retrieved_document_ids=list(
            dict.fromkeys(
                hit.get("document_id", "")
                for hit in payload.get("retrieval_hits", [])[:5]
                if hit.get("document_id")
            )
        ),
        cited_chunk_ids=[source["chunk_id"] for source in payload.get("sources", [])],
        route=payload.get("route"),
        strategy=payload.get("strategy"),
        retry_count=payload.get("retry_count", 0),
        conflict_detected=bool(payload.get("conflict", False)),
        terminated=termination is not None,
        abstained=is_abstention_termination(termination),
        latency_seconds=time.perf_counter() - started,
        llm_calls=model.calls - started_calls,
        retrieval_rounds=sum(event.get("stage") == "retrieve" for event in trace),
        answer=str(payload.get("answer", "")),
        evidence_status=payload.get("evidence_status"),
        subquery_specs=payload.get("subquery_specs", []),
        rewritten_subqueries=payload.get("rewritten_subqueries", []),
        supported_subquery_ids=payload.get("supported_subquery_ids", []),
        relevant_labels=payload.get("relevant_labels", []),
        termination_reason=termination,
        validation_violations=validation.get("violations", []),
        initial_validation_violations=validation.get("initial_violations", []),
        repair_validation_violations=validation.get("repair_violations", []),
        retrieved_evidence=map_retrieved_evidence(payload.get("retrieval_hits", [])),
        public_trace=[
            {
                key: event.get(key)
                for key in (
                    "stage",
                    "decision",
                    "retrieved_count",
                    "fused_count",
                    "selected_count",
                    "retry_count",
                    "llm_calls",
                    "termination",
                    "duration_ms",
                )
            }
            for event in trace
        ],
    )
    result.failure_labels = failure_labels(case, result)
    return result


def is_abstention_termination(termination: str | None) -> bool:
    return termination in {
        "unsupported",
        "out_of_scope",
        "clarification",
        "retry_noop",
        "validation_failed",
    }


def normalize_model_name(name: str) -> str:
    """Return the exact Ollama identifier, adding the implicit latest tag."""
    normalized = name.strip()
    return normalized if ":" in normalized else f"{normalized}:latest"


def required_models_for_systems(  # lanorme: ignore[KWARG-001] -- Public API supports positional model selection
    systems: Sequence[SystemName], chat_model: str | None = None
) -> tuple[str, ...]:
    """Return only the local models needed by the selected evaluation systems."""
    required: list[str] = []
    if any(system != "bm25" for system in systems):
        required.append(normalize_model_name(config.embedding_model))
    if any(system in ANSWER_SYSTEMS for system in systems):
        required.append(normalize_model_name(chat_model or config.llm_model))
    return tuple(dict.fromkeys(required))


def _require_ollama(required_models: Sequence[str] | None = None) -> None:
    try:
        with urllib.request.urlopen(f"{config.ollama_base_url}/api/tags", timeout=3) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Live evaluation requires Ollama at {config.ollama_base_url}. "
            "Start Ollama and pull the configured chat and embedding models."
        ) from exc
    available = {
        normalize_model_name(str(item.get("name", "")))
        for item in payload.get("models", [])
        if item.get("name")
    }
    configured = required_models or (config.llm_model, config.embedding_model)
    required = {normalize_model_name(name) for name in configured}
    missing = sorted(required - available)
    if missing:
        commands = ", ".join(f"ollama pull {name}" for name in missing)
        raise RuntimeError(
            f"Missing required Ollama model(s): {', '.join(missing)}. Run: {commands}"
        )


def multihop_settings() -> Settings:
    runtime = PROJECT_ROOT / "evals" / "runtime"
    return Settings(
        sources_dir=MULTIHOP_ROOT / "corpus",
        data_dir=runtime,
        chroma_dir=runtime / "chroma",
        manifest_path=runtime / "manifest.json",
        trace_dir=runtime / "traces",
        logs_dir=runtime / "logs",
    )


def _evidence_matches(evidence: str, chunk: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    return normalize(evidence) in normalize(chunk)


def preflight_multihop(
    cases: Sequence[EvaluationCase],
    source_map_path: Path = MULTIHOP_ROOT / "source_map.json",
    manager: VectorDBManager | None = None,
    *,
    check_models: bool = True,
) -> list[EvaluationCase]:
    """Resolve stable benchmark evidence before any live-model invocation."""
    cases_path = source_map_path.with_name("cases.jsonl")
    if not source_map_path.exists() or not cases_path.exists():
        raise RuntimeError(
            "MultiHop-RAG benchmark files are missing. Run: "
            "uv run python scripts/prepare_multihop_eval.py --index"
        )
    payload = json.loads(source_map_path.read_text(encoding="utf-8"))
    source_map = payload.get("documents", payload)
    active_manager = manager or VectorDBManager(multihop_settings())
    manifest = active_manager.manifest()
    if not manifest.documents or active_manager.chunk_count() == 0:
        raise RuntimeError(
            "MultiHop-RAG manifest and Chroma must be populated. Run the preparation script "
            "with --index."
        )
    dev_ids = {case.id for case in cases if case.split == "development"}
    test_ids = {case.id for case in cases if case.split == "test"}
    if dev_ids & test_ids:
        raise RuntimeError("MultiHop-RAG development and held-out IDs overlap.")
    collection = active_manager.setup().get(include=["documents", "metadatas"])
    chunk_rows = list(
        zip(
            collection.get("ids", []),
            collection.get("documents", []) or [],
            collection.get("metadatas", []) or [],
            strict=True,
        )
    )
    resolved: list[EvaluationCase] = []
    failures: list[str] = []
    for case in cases:
        chunk_ids: list[str] = []
        document_ids: list[str] = []
        for evidence in case.gold_evidence:
            record = source_map.get(evidence.benchmark_document_id)
            if not record:
                failures.append(f"{case.id}: unknown document {evidence.benchmark_document_id}")
                continue
            document_id = str(record["document_id"])
            matches = [
                str(chunk_id)
                for chunk_id, text, metadata in chunk_rows
                if str(metadata.get("document_id", "")) == document_id
                and _evidence_matches(evidence.evidence_text, str(text))
            ]
            if not matches:
                failures.append(
                    f"{case.id}: evidence not found in indexed document "
                    f"{evidence.benchmark_document_id}"
                )
                continue
            document_ids.append(document_id)
            chunk_ids.extend(matches)
        resolved.append(
            case.model_copy(
                update={
                    "relevant_chunk_ids": list(dict.fromkeys(chunk_ids)),
                    "relevant_document_ids": list(dict.fromkeys(document_ids)),
                }
            )
        )
    if failures:
        raise RuntimeError("Gold evidence resolution failed:\n- " + "\n- ".join(failures))
    if check_models:
        _require_ollama()
    return resolved


def run_evaluation(  # lanorme: ignore[PARAM-001,SIZE-002,COMPLEXITY-001,KWARG-001] -- Stable public evaluator API
    dataset: Path,
    systems: Sequence[SystemName],
    split: Split,
    *,
    dataset_name: str = "custom",
    chat_model: str | None = None,
    case_ids: set[str] | None = None,
    case_timeout_seconds: float = 30.0,
) -> Path:
    selected = tuple(dict.fromkeys(systems))
    if not selected:
        raise ValueError("Select at least one evaluation system.")
    selected_chat_model = normalize_model_name(chat_model or config.llm_model)
    raw = dataset.read_bytes()
    all_cases = load_cases(dataset)
    if dataset_name == "multihop":
        manager = VectorDBManager(multihop_settings())
        all_cases = preflight_multihop(all_cases, manager=manager, check_models=False)
    else:
        manager = VectorDBManager()
    required_models = required_models_for_systems(selected, selected_chat_model)
    if required_models:
        _require_ollama(required_models)
    cases = filter_cases(all_cases, split, case_ids)
    if not cases:
        raise ValueError(f"Dataset has no cases for split '{split}'")
    retriever = Retriever(manager.setup())
    counted: CountingModel | None = None
    graph: RAGGraph | None = None
    if any(system in ANSWER_SYSTEMS for system in selected):
        counted = CountingModel(
            ChatOllama(
                model=selected_chat_model,
                base_url=config.ollama_base_url,
                temperature=config.temperature,
                num_predict=512,
                client_kwargs={"timeout": CANONICAL_REQUEST_TIMEOUT_SECONDS},
            )
        )
        if FULL_RAG_SYSTEM in selected:
            graph = RAGGraph(manager, llm=counted)  # type: ignore[arg-type]
    results: list[CaseResult] = []
    for system in selected:
        for case in cases:
            if system == FULL_RAG_SYSTEM:
                result = run_agentic_case(
                    case,
                    graph,  # type: ignore[arg-type]
                    counted,  # type: ignore[arg-type]
                    timeout_seconds=case_timeout_seconds,
                )
            elif system in FIXED_RAG_SYSTEMS:
                result = run_fixed_rag_case(
                    case,
                    system,
                    retriever,
                    counted,  # type: ignore[arg-type]
                )
            else:
                result = run_retrieval_case(case, system, retriever)
            results.append(result)
    metrics = {
        system: aggregate_metrics(
            cases,
            [item for item in results if item.system == system],
            system=system,
        )
        for system in selected
    }
    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-{split}"
    experiment = ExperimentConfig(
        run_id=run_id,
        timestamp=now.isoformat(),
        git_commit=_git_commit(),
        dataset_hash=hashlib.sha256(raw).hexdigest(),
        evaluated_split=split,
        systems=list(selected),
        chat_model=selected_chat_model,
        embedding_model=config.embedding_model,
        temperature=config.temperature,
        fixed_rag_prompt_id=FIXED_RAG_PROMPT_ID,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        retrieval_limit=5,
        semantic_candidates=config.semantic_candidates,
        sparse_candidates=config.sparse_candidates,
        retry_limit=config.max_retries,
        subquery_limit=config.max_subqueries,
        case_timeout_seconds=case_timeout_seconds,
        dataset_name=dataset_name,
        dataset_version=("yixuantt/MultiHopRAG" if dataset_name == "multihop" else "local"),
        dataset_license=("ODC-By-1.0" if dataset_name == "multihop" else "unspecified"),
    )
    return write_experiment(
        PROJECT_ROOT / "evals" / "results" / dataset_name, experiment, results, metrics
    )


def _parse_systems(value: str) -> list[SystemName]:
    requested = list(SYSTEMS) if value == "all" else [item.strip() for item in value.split(",")]
    invalid = sorted(set(requested) - set(SYSTEMS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown system(s): {', '.join(invalid)}")
    return list(dict.fromkeys(requested))  # type: ignore[return-value]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=_parse_systems, default=list(SYSTEMS))
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--dataset", default="multihop")
    parser.add_argument("--model", default=config.llm_model)
    parser.add_argument(
        "--case-timeout-seconds",
        type=_positive_float,
        default=30.0,
        help="Hard wall-clock deadline for each agentic case (default: 30).",
    )
    parser.add_argument(
        "--case-ids",
        help="Comma-separated case IDs for a bounded diagnostic run.",
    )
    args = parser.parse_args()
    try:
        dataset_name = str(args.dataset)
        if dataset_name == "multihop":
            dataset = MULTIHOP_ROOT / "cases.jsonl"
        elif dataset_name in {"regression", "mvp"}:
            dataset = PROJECT_ROOT / "evals" / "mvp_cases.jsonl"
            dataset_name = "regression"
        else:
            dataset = Path(dataset_name)
            dataset_name = "custom"
        run_kwargs = {
            "dataset_name": dataset_name,
            "chat_model": args.model,
            "case_timeout_seconds": args.case_timeout_seconds,
        }
        if args.case_ids:
            run_kwargs["case_ids"] = {
                item.strip() for item in args.case_ids.split(",") if item.strip()
            }
        output = run_evaluation(dataset, args.systems, args.split, **run_kwargs)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"evaluation error: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
