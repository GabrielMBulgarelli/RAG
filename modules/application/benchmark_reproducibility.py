"""Full RAG Benchmark reproducibility contracts and validation."""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field, JsonValue

from modules.config import Settings
from modules.evaluation import dataset_content_hash
from modules.evaluation_models import (
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
    EvaluationCase,
)


class BenchmarkGraphConfiguration(BaseModel):
    max_candidates: int = Field(ge=1)
    maximum_context_chunks: int = Field(ge=1)
    retry_limit: int = Field(ge=0)
    subquery_limit: int = Field(ge=1)


class BenchmarkReproducibility(BaseModel):
    benchmark_name: str
    git_commit: str
    dataset_identifier: str
    dataset_hash: str
    case_ids: list[str]
    expected_result_count: int = Field(ge=0)
    chat_model: str
    embedding_model: str
    temperature: float
    fixed_rag_prompt_id: str
    graph_configuration: BenchmarkGraphConfiguration
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    retrieval_limit: int = Field(ge=1)
    semantic_candidates: int = Field(ge=1)
    sparse_candidates: int = Field(ge=1)
    maximum_context_chunks: int = Field(ge=1)
    retry_limit: int = Field(ge=0)
    subquery_limit: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)


@dataclass(frozen=True)
class BenchmarkRuntimeIdentity:
    git_commit: str
    chat_model: str
    embedding_model: str


def _ensure_canonical_configuration(
    *, values: dict[str, JsonValue], graph: BenchmarkGraphConfiguration
) -> None:
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
    expected_graph = {
        "max_candidates": 20,
        "maximum_context_chunks": CANONICAL_MAX_CONTEXT_CHUNKS,
        "retry_limit": CANONICAL_RETRY_LIMIT,
        "subquery_limit": CANONICAL_SUBQUERY_LIMIT,
    }
    if graph.model_dump(mode="json") != expected_graph or any(
        values[name] != expected_value for name, expected_value in expected.items()
    ):
        raise ValueError("The Full RAG Benchmark requires the canonical benchmark configuration.")


def canonical_reproducibility(
    *,
    settings: Settings,
    cases: Sequence[EvaluationCase],
    identity: BenchmarkRuntimeIdentity,
) -> dict[str, JsonValue]:
    graph = BenchmarkGraphConfiguration(
        max_candidates=settings.max_candidates,
        maximum_context_chunks=settings.max_context_chunks,
        retry_limit=settings.max_retries,
        subquery_limit=settings.max_subqueries,
    )
    values = BenchmarkReproducibility(
        benchmark_name="full_rag_benchmark",
        git_commit=identity.git_commit,
        dataset_identifier="yixuantt/MultiHopRAG",
        dataset_hash=dataset_content_hash(cases),
        case_ids=list(CANONICAL_BENCHMARK_CASE_IDS),
        expected_result_count=CANONICAL_BENCHMARK_RESULT_COUNT,
        chat_model=identity.chat_model,
        embedding_model=identity.embedding_model,
        temperature=settings.temperature,
        fixed_rag_prompt_id=FIXED_RAG_PROMPT_ID,
        graph_configuration=graph,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        retrieval_limit=CANONICAL_RETRIEVAL_LIMIT,
        semantic_candidates=settings.semantic_candidates,
        sparse_candidates=settings.sparse_candidates,
        maximum_context_chunks=settings.max_context_chunks,
        retry_limit=settings.max_retries,
        subquery_limit=settings.max_subqueries,
        request_timeout_seconds=CANONICAL_REQUEST_TIMEOUT_SECONDS,
    ).model_dump(mode="json")
    _ensure_canonical_configuration(values=values, graph=graph)
    return values
