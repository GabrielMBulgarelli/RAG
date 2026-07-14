"""Small comparative evaluation harness for the four MVP RAG systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from modules.config import PROJECT_ROOT, Settings, config
from modules.models import RetrievalHit
from modules.rag_graph import RAGGraph
from modules.retrieval import Retriever, reciprocal_rank_fusion
from modules.vector_db import VectorDBManager

Split = Literal["development", "test"]
SystemName = Literal["dense", "bm25", "hybrid", "agentic"]
SYSTEMS: tuple[SystemName, ...] = ("dense", "bm25", "hybrid", "agentic")
FAILURE_ORDER = (
    "route_error",
    "strategy_error",
    "retrieval_miss",
    "invalid_citation",
    "citation_coverage_miss",
    "over_abstention",
    "failed_abstention",
    "retry_error",
    "conflict_miss",
    "non_termination",
    "runtime_error",
)

MULTIHOP_ROOT = PROJECT_ROOT / "evals" / "multihop"


class BenchmarkEvidence(BaseModel):
    benchmark_document_id: str
    source: str = ""
    title: str = ""
    url: str = ""
    evidence_text: str
    author: str = ""
    category: str = ""
    published_at: str = ""


class EvaluationCase(BaseModel):
    id: str
    split: Split
    category: str
    question: str
    answerable: bool
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    gold_evidence: list[BenchmarkEvidence] = Field(default_factory=list)
    expected_answer: str | None = None
    expected_route: str
    expected_strategy: str
    expected_retry: bool
    expected_conflict: bool


class CaseResult(BaseModel):
    case_id: str
    system: SystemName
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    route: str | None = None
    strategy: str | None = None
    retry_count: int = Field(default=0, ge=0)
    conflict_detected: bool = False
    terminated: bool = False
    abstained: bool = False
    latency_seconds: float = Field(default=0.0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    retrieval_rounds: int = Field(default=0, ge=0)
    failure_labels: list[str] = Field(default_factory=list)
    runtime_error: str | None = None
    answer: str = ""


class ExperimentConfig(BaseModel):
    run_id: str
    timestamp: str
    git_commit: str
    dataset_hash: str
    evaluated_split: Split
    systems: list[SystemName]
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_limit: int
    semantic_candidates: int
    sparse_candidates: int
    retry_limit: int
    subquery_limit: int
    dataset_name: str = "custom"
    dataset_version: str = "local"
    dataset_license: str = "unspecified"


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    for rank, chunk_id in enumerate(retrieved[:k], 1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 1.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], 1)
        if chunk_id in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])


def citation_precision(cited: Sequence[str], retrieved: Sequence[str]) -> float:
    if not cited:
        return 1.0
    return len(set(cited) & set(retrieved)) / len(set(cited))


def gold_citation_coverage(cited: Sequence[str], relevant: Sequence[str]) -> float:
    """Fraction of expected relevant chunk IDs cited (not claim-level coverage)."""
    gold = set(relevant)
    if not gold:
        return 1.0
    return len(set(cited) & gold) / len(gold)


def _normalize_answer(value: str) -> str:
    value = value.lower().translate(str.maketrans({char: " " for char in string.punctuation}))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def normalized_exact_match(prediction: str, expected: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(expected))


def token_f1(prediction: str, expected: str) -> float:
    predicted = _normalize_answer(prediction).split()
    gold = _normalize_answer(expected).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _safe_ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    return numerator / denominator if denominator else empty


def aggregate_metrics(
    cases: Sequence[EvaluationCase], results: Sequence[CaseResult]
) -> dict[str, float]:
    by_id = {item.id: item for item in cases}
    paired = [(by_id[item.case_id], item) for item in results if item.case_id in by_id]
    retrieval = [(case, result) for case, result in paired if case.relevant_chunk_ids]
    document_retrieval = [(case, result) for case, result in paired if case.relevant_document_ids]
    agentic = [(case, result) for case, result in paired if result.system == "agentic"]

    def mean(values: Iterable[float]) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized) if materialized else 0.0

    route = [(case, result) for case, result in agentic if result.route is not None]
    strategy = [(case, result) for case, result in agentic if result.strategy is not None]
    retry_tp = sum(case.expected_retry and result.retry_count > 0 for case, result in agentic)
    retry_fp = sum(not case.expected_retry and result.retry_count > 0 for case, result in agentic)
    retry_fn = sum(case.expected_retry and result.retry_count == 0 for case, result in agentic)
    retry_expected = retry_tp + retry_fn
    retry_predicted = retry_tp + retry_fp
    retry_precision_empty = 0.0 if retry_expected else 1.0

    chunk_recall = mean(
        recall_at_k(result.retrieved_chunk_ids, set(case.relevant_chunk_ids))
        for case, result in retrieval
    )
    answer_pairs = [
        (case, result)
        for case, result in agentic
        if case.expected_answer is not None and case.answerable
    ]
    return {
        "recall_at_5": chunk_recall,
        "chunk_recall_at_5": chunk_recall,
        "document_recall_at_5": mean(
            recall_at_k(result.retrieved_document_ids, set(case.relevant_document_ids))
            for case, result in document_retrieval
        ),
        "mrr_at_5": mean(
            mrr_at_k(result.retrieved_chunk_ids, set(case.relevant_chunk_ids))
            for case, result in retrieval
        ),
        "ndcg_at_5": mean(
            ndcg_at_k(result.retrieved_chunk_ids, set(case.relevant_chunk_ids))
            for case, result in retrieval
        ),
        "route_accuracy": _safe_ratio(
            sum(result.route == case.expected_route for case, result in route), len(route)
        ),
        "strategy_accuracy": _safe_ratio(
            sum(result.strategy == case.expected_strategy for case, result in strategy),
            len(strategy),
        ),
        "retry_precision": _safe_ratio(retry_tp, retry_predicted, empty=retry_precision_empty),
        "retry_recall": _safe_ratio(retry_tp, retry_expected),
        "conflict_accuracy": _safe_ratio(
            sum(result.conflict_detected == case.expected_conflict for case, result in agentic),
            len(agentic),
        ),
        "termination_rate": _safe_ratio(
            sum(result.terminated for _, result in paired), len(paired)
        ),
        "citation_precision": mean(
            citation_precision(result.cited_chunk_ids, result.retrieved_chunk_ids)
            for _, result in agentic
        ),
        "gold_evidence_citation_coverage": mean(
            gold_citation_coverage(result.cited_chunk_ids, case.relevant_chunk_ids)
            for case, result in agentic
        ),
        "abstention_accuracy": _safe_ratio(
            sum(result.abstained == (not case.answerable) for case, result in agentic),
            len(agentic),
        ),
        "mean_latency_seconds": mean(result.latency_seconds for _, result in paired),
        "p95_latency_seconds": p95([result.latency_seconds for _, result in paired]),
        "mean_llm_calls_per_query": mean(float(result.llm_calls) for _, result in paired),
        "mean_retrieval_rounds_per_query": mean(
            float(result.retrieval_rounds) for _, result in paired
        ),
        "normalized_answer_exact_match": mean(
            normalized_exact_match(result.answer, case.expected_answer or "")
            for case, result in answer_pairs
        ),
        "answer_token_f1": mean(
            token_f1(result.answer, case.expected_answer or "") for case, result in answer_pairs
        ),
    }


def failure_labels(case: EvaluationCase, result: CaseResult) -> list[str]:
    labels: set[str] = set()
    if result.route is not None and result.route != case.expected_route:
        labels.add("route_error")
    if result.strategy is not None and result.strategy != case.expected_strategy:
        labels.add("strategy_error")
    if case.relevant_chunk_ids and not (
        set(result.retrieved_chunk_ids) & set(case.relevant_chunk_ids)
    ):
        labels.add("retrieval_miss")
    if set(result.cited_chunk_ids) - set(result.retrieved_chunk_ids):
        labels.add("invalid_citation")
    if gold_citation_coverage(result.cited_chunk_ids, case.relevant_chunk_ids) < 1:
        labels.add("citation_coverage_miss")
    if case.answerable and result.abstained:
        labels.add("over_abstention")
    if not case.answerable and not result.abstained:
        labels.add("failed_abstention")
    if (result.retry_count > 0) != case.expected_retry:
        labels.add("retry_error")
    if result.conflict_detected != case.expected_conflict:
        labels.add("conflict_miss")
    if not result.terminated:
        labels.add("non_termination")
    if result.runtime_error:
        labels.add("runtime_error")
    return [label for label in FAILURE_ORDER if label in labels]


def filter_cases(cases: Sequence[EvaluationCase], split: Split) -> list[EvaluationCase]:
    return [case for case in cases if case.split == split]


class _CallState:
    calls = 0


class CountingModel:
    """Thin proxy counting all calls, including structured-output calls, at one boundary."""

    def __init__(self, model: Any, state: _CallState | None = None):
        self._model = model
        self._state = state or _CallState()

    @property
    def calls(self) -> int:
        return self._state.calls

    def invoke(self, value: object, **kwargs: object) -> Any:
        self._state.calls += 1
        return self._model.invoke(value, **kwargs)

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


def run_agentic_case(case: EvaluationCase, graph: RAGGraph, model: CountingModel) -> CaseResult:
    started_calls = model.calls
    started = time.perf_counter()
    try:
        payload = graph.process_query(case.question, f"evaluation-{case.id}")
    except Exception as exc:  # noqa: BLE001 - evaluation must preserve per-case failures
        result = CaseResult(
            case_id=case.id,
            system="agentic",
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
    result = CaseResult(
        case_id=case.id,
        system="agentic",
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
        conflict_detected=False,
        terminated=termination is not None,
        abstained=termination in {"unsupported", "out_of_scope", "clarification"},
        latency_seconds=time.perf_counter() - started,
        llm_calls=model.calls - started_calls,
        retrieval_rounds=sum(event.get("stage") == "retrieve" for event in trace),
        answer=str(payload.get("answer", "")),
    )
    result.failure_labels = failure_labels(case, result)
    return result


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def write_experiment(
    results_root: Path,
    experiment: ExperimentConfig,
    results: Sequence[CaseResult],
    metrics: dict[str, dict[str, float]],
) -> Path:
    output = results_root / experiment.run_id
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")
    summary = {"configuration": experiment.model_dump(mode="json"), "metrics": metrics}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Evaluation {experiment.run_id}", "", f"Split: `{experiment.evaluated_split}`", ""]
    for system, values in metrics.items():
        lines.extend([f"## {system}", ""])
        lines.extend(f"- {name}: {value:.6f}" for name, value in values.items())
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def _require_ollama() -> None:
    try:
        with urllib.request.urlopen(f"{config.ollama_base_url}/api/tags", timeout=3) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Live evaluation requires Ollama at {config.ollama_base_url}. "
            "Start Ollama and pull the configured chat and embedding models."
        ) from exc
    available = {item.get("name", "").split(":")[0] for item in payload.get("models", [])}
    required = {config.llm_model.split(":")[0], config.embedding_model.split(":")[0]}
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


def run_evaluation(
    dataset: Path, systems: Sequence[SystemName], split: Split, *, dataset_name: str = "custom"
) -> Path:
    raw = dataset.read_bytes()
    all_cases = load_cases(dataset)
    if dataset_name == "multihop":
        manager = VectorDBManager(multihop_settings())
        all_cases = preflight_multihop(all_cases, manager=manager)
    else:
        _require_ollama()
        manager = VectorDBManager()
    cases = filter_cases(all_cases, split)
    if not cases:
        raise ValueError(f"Dataset has no cases for split '{split}'")
    retriever = Retriever(manager.setup())
    counted = CountingModel(
        ChatOllama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
        )
    )
    graph = RAGGraph(manager, llm=counted)  # type: ignore[arg-type]
    results: list[CaseResult] = []
    for system in systems:
        for case in cases:
            results.append(
                run_agentic_case(case, graph, counted)
                if system == "agentic"
                else run_retrieval_case(case, system, retriever)
            )
    metrics = {
        system: aggregate_metrics(cases, [item for item in results if item.system == system])
        for system in systems
    }
    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-{split}"
    experiment = ExperimentConfig(
        run_id=run_id,
        timestamp=now.isoformat(),
        git_commit=_git_commit(),
        dataset_hash=hashlib.sha256(raw).hexdigest(),
        evaluated_split=split,
        systems=list(systems),
        chat_model=config.llm_model,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        retrieval_limit=5,
        semantic_candidates=config.semantic_candidates,
        sparse_candidates=config.sparse_candidates,
        retry_limit=config.max_retries,
        subquery_limit=config.max_subqueries,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=_parse_systems, default=list(SYSTEMS))
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--dataset", default="regression")
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
        output = run_evaluation(dataset, args.systems, args.split, dataset_name=dataset_name)
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"evaluation error: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
