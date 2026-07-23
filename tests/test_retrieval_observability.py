from types import SimpleNamespace
from typing import Any, cast

from langchain_core.documents import Document

from modules import rag_graph
from modules.models import (
    EvidenceGrade,
    EvidenceStatus,
    RetrievalBatch,
    RetrievalHit,
    RetrievalStrategy,
    Route,
)
from modules.rag_graph import RAGGraph
from modules.retrieval import Retriever, reciprocal_rank_fusion, select_candidates


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
        self.semantic_limits: list[int] = []
        self.get_calls = 0

    def similarity_search_with_relevance_scores(self, query, *, k, filter=None):
        self.semantic_filter = filter
        self.semantic_limits.append(k)
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
        self.get_calls += 1
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


def test_search_uses_separate_budgets_and_reports_each_retrieval_stage() -> None:
    store = FakeVectorStore()
    retriever = Retriever(store)

    result = retriever.search(
        "rare keyword",
        strategy="hybrid",
        semantic_k=10,
        sparse_k=10,
        fusion_limit=20,
        selection_limit=6,
    )

    assert store.semantic_limits == [10]
    assert result.retrieved_count == 2
    assert result.fused_count == 2
    assert result.selected_count == 2
    assert all(item.selection_score is not None for item in result.hits)


def test_sparse_index_is_reused_until_explicitly_invalidated() -> None:
    store = FakeVectorStore()
    retriever = Retriever(store)

    retriever.sparse("rare", 10)
    retriever.sparse("common", 10)
    assert store.get_calls == 1

    retriever.invalidate_sparse_index()
    retriever.sparse("general", 10)
    assert store.get_calls == 2


def test_selection_rewards_subquery_coverage_and_document_diversity() -> None:
    candidates = [
        RetrievalHit(
            chunk_id="best",
            document_id="doc-a",
            content="warranty coverage and repair terms",
            filename="a.txt",
            page=1,
            score=0.03,
            fused_score=0.03,
            subqueries=["warranty"],
        ),
        RetrievalHit(
            chunk_id="redundant",
            document_id="doc-a",
            content="warranty coverage and repair terms apply",
            filename="a.txt",
            page=2,
            score=0.029,
            fused_score=0.029,
            subqueries=["warranty"],
        ),
        RetrievalHit(
            chunk_id="complementary",
            document_id="doc-b",
            content="refund window is thirty calendar days",
            filename="b.txt",
            page=1,
            score=0.028,
            fused_score=0.028,
            subqueries=["refund period"],
        ),
    ]

    selected = select_candidates(candidates, limit=2)

    assert [item.chunk_id for item in selected] == ["best", "complementary"]
    assert all(item.selection_score is not None for item in selected)


def test_retrieve_records_provenance_counts_and_duration() -> None:
    graph = RAGGraph.__new__(RAGGraph)
    graph.retriever = cast(
        Any,
        SimpleNamespace(
            search=lambda query, **_kwargs: RetrievalBatch(
                hits=[hit("shared", 1.0, query=query)],
                retrieved_count=2,
                fused_count=1,
                selected_count=1,
            )
        ),
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
    assert event.candidate_count == 4
    assert event.retrieved_count == 4
    assert event.fused_count == 1
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
        "grade": EvidenceGrade(
            status=EvidenceStatus.INSUFFICIENT,
            answer_supported=False,
        ),
        "queries": ["refined query"],
        "trace": [],
    }

    result = graph._validate(cast(Any, state))["result"]

    assert result.subqueries == ["refined query"]
    assert result.trace[-1].stage == "terminate"
    assert result.trace[-1].termination == "unsupported"
