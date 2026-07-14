from types import SimpleNamespace
from typing import Any, cast

from langchain_core.documents import Document

from modules import rag_graph
from modules.models import (
    EvidenceGrade,
    EvidenceStatus,
    RetrievalHit,
    RetrievalStrategy,
    Route,
)
from modules.rag_graph import RAGGraph
from modules.retrieval import Retriever, reciprocal_rank_fusion


def hit(chunk_id: str, score: float, *, query: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        filename="guide.txt",
        page=1,
        score=score,
        subqueries=[query],
    )


class FakeVectorStore:
    def __init__(self) -> None:
        self.semantic_filter = None
        self.sparse_filter = None

    def similarity_search_with_relevance_scores(self, query, *, k, filter=None):
        self.semantic_filter = filter
        return [
            (
                Document(
                    page_content="semantic result",
                    metadata={
                        "chunk_id": "semantic",
                        "document_id": "doc-1",
                        "filename": "guide.txt",
                        "page": 2,
                    },
                ),
                0.8,
            )
        ]

    def get(self, *, include, where=None):
        self.sparse_filter = where
        return {
            "documents": ["rare keyword", "common handbook", "general policy"],
            "metadatas": [
                {
                    "chunk_id": "sparse",
                    "document_id": "doc-1",
                    "filename": "guide.txt",
                    "page": 3,
                },
                {"chunk_id": "other-1", "filename": "guide.txt", "page": 4},
                {"chunk_id": "other-2", "filename": "guide.txt", "page": 5},
            ],
        }


def test_follow_up_detection_does_not_rewrite_independent_questions() -> None:
    assert rag_graph.is_contextual_follow_up("What about its warranty?")
    assert rag_graph.is_contextual_follow_up("And how long does that last?")
    assert not rag_graph.is_contextual_follow_up("What is the annual leave policy?")


def test_graph_compiles_without_node_state_key_collisions() -> None:
    graph = RAGGraph.__new__(RAGGraph)

    assert graph._compile() is not None


def test_hybrid_preserves_component_scores_and_subquery_provenance() -> None:
    semantic = [hit("both", 0.9, query="semantic query"), hit("dense", 0.7, query="q")]
    sparse = [hit("both", 4.2, query="keyword query"), hit("sparse", 3.0, query="q")]

    fused = reciprocal_rank_fusion(semantic, sparse, limit=3)
    by_id = {item.chunk_id: item for item in fused}

    assert by_id["both"].semantic_score == 0.9
    assert by_id["both"].sparse_score == 4.2
    assert by_id["both"].fused_score == by_id["both"].score
    assert by_id["both"].subqueries == ["keyword query", "semantic query"]


def test_filters_reach_dense_and_sparse_retrieval() -> None:
    store = FakeVectorStore()
    retriever = Retriever(store)
    filters = {"filename": "guide.txt"}

    dense = retriever.semantic("warranty", 5, filters=filters)
    sparse = retriever.sparse("keyword", 5, filters=filters)

    assert store.semantic_filter == filters
    assert store.sparse_filter == filters
    assert dense[0].semantic_score == 0.8
    assert sparse[0].sparse_score is not None
    assert dense[0].document_id == "doc-1"


def test_retrieve_records_provenance_counts_and_duration() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    graph.retriever = cast(
        Any,
        SimpleNamespace(search=lambda query, **_kwargs: [hit("shared", 1.0, query=query)]),
    )
    state = {
        "queries": ["first", "second"],
        "rewritten_query": "original",
        "strategy": RetrievalStrategy.HYBRID,
        "retry_count": 0,
        "trace": [],
        "filters": {"document_id": "doc-1"},
    }

    update = graph._retrieve(cast(Any, state))

    assert update["hits"][0].subqueries == ["first", "second"]
    event = update["trace"][-1]
    assert event.stage == "retrieve"
    assert event.candidate_count == 2
    assert event.selected_count == 1
    assert event.duration_ms >= 0


def test_validate_exposes_subqueries_and_public_termination_event() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    state = {
        "answer": "No supported answer.",
        "sources": [],
        "route": Route.SIMPLE_SEARCH,
        "strategy": RetrievalStrategy.SEMANTIC,
        "retry_count": 1,
        "grade": EvidenceGrade(status=EvidenceStatus.INSUFFICIENT),
        "queries": ["refined query"],
        "trace": [],
    }

    result = graph._validate(cast(Any, state))["result"]

    assert result.subqueries == ["refined query"]
    assert result.trace[-1].stage == "terminate"
    assert result.trace[-1].termination == "unsupported"
