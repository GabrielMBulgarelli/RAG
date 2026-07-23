"""Semantic and BM25 retrieval with reciprocal-rank fusion."""

import re
from dataclasses import dataclass
from typing import NotRequired, TypedDict, Unpack

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .contracts import Metadata, RetrievalVectorStore, VectorFilter
from .models import RetrievalBatch, RetrievalHit


@dataclass(frozen=True)
class _SparseIndex:
    documents: list[str]
    metadatas: list[Metadata]
    bm25: BM25Okapi


@dataclass(frozen=True)
class SearchOptions:
    strategy: str
    semantic_k: int
    sparse_k: int
    fusion_limit: int
    selection_limit: int
    filters: dict[str, str] | None = None


class SearchArguments(TypedDict):
    strategy: str
    semantic_k: int
    sparse_k: int
    fusion_limit: int
    selection_limit: int
    filters: NotRequired[dict[str, str] | None]


class FusionArguments(TypedDict):
    sparse: NotRequired[list[RetrievalHit]]
    limit: int
    rank_constant: NotRequired[int]


@dataclass
class _SelectionState:
    remaining: list[RetrievalHit]
    selected: list[RetrievalHit]
    covered_queries: set[str]
    covered_documents: set[str]


def _candidate_score(
    *,
    item: RetrievalHit,
    state: _SelectionState,
    relevance: dict[str, float],
    query_count: int,
) -> float:
    new_queries = set(item.subqueries) - state.covered_queries
    coverage_bonus = 0.2 * len(new_queries) / max(query_count, 1)
    document_key = item.document_id or item.filename
    diversity_bonus = 0.1 if document_key not in state.covered_documents else 0.0
    redundancy = max(
        (
            _lexical_similarity(first=item.content, second=chosen.content)
            for chosen in state.selected
        ),
        default=0.0,
    )
    return 0.7 * relevance[item.chunk_id] + coverage_bonus + diversity_bonus - 0.25 * redundancy


def _choose_candidate(
    *,
    state: _SelectionState,
    relevance: dict[str, float],
    query_count: int,
    preserved: set[str],
) -> RetrievalHit:
    scored = [
        (
            item.chunk_id in preserved,
            _candidate_score(item=item, state=state, relevance=relevance, query_count=query_count),
            relevance[item.chunk_id],
            item.chunk_id,
            item,
        )
        for item in state.remaining
    ]
    _, score, _, _, chosen = min(
        scored, key=lambda value: (-value[0], -value[1], -value[2], value[3])
    )
    return chosen.model_copy(update={"selection_score": score})


def _rank_candidates(candidates: list[RetrievalHit]) -> list[RetrievalHit]:
    return sorted(
        candidates,
        key=lambda item: (
            -(item.fused_score if item.fused_score is not None else item.score),
            item.chunk_id,
        ),
    )


def _select_from_ordered(
    *, ordered: list[RetrievalHit], limit: int, preserved: set[str]
) -> list[RetrievalHit]:
    relevance = {item.chunk_id: 1.0 - (rank / len(ordered)) for rank, item in enumerate(ordered)}
    all_queries = {query for item in ordered for query in item.subqueries}
    state = _SelectionState(list(ordered), [], set(), set())
    while state.remaining and len(state.selected) < limit:
        chosen = _choose_candidate(
            state=state,
            relevance=relevance,
            query_count=len(all_queries),
            preserved=preserved,
        )
        state.selected.append(chosen)
        state.remaining = [item for item in state.remaining if item.chunk_id != chosen.chunk_id]
        state.covered_queries.update(chosen.subqueries)
        state.covered_documents.add(chosen.document_id or chosen.filename)
    return state.selected


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.lower())


def _hit(*, document: Document, score: float, score_type: str) -> RetrievalHit:
    metadata = document.metadata
    return RetrievalHit(
        chunk_id=str(metadata.get("chunk_id", "")),
        content=document.page_content,
        document_id=str(metadata.get("document_id", "")),
        filename=str(metadata.get("filename", "Unknown")),
        page=int(metadata.get("page", 1)),
        score=float(score),
        semantic_score=float(score) if score_type == "semantic" else None,
        sparse_score=float(score) if score_type == "sparse" else None,
    )


def _where(filters: dict[str, str] | None) -> VectorFilter | None:
    if not filters:
        return None
    clauses: list[VectorFilter] = [{key: value} for key, value in sorted(filters.items())]
    return clauses[0] if len(clauses) == 1 else {"$and": list(clauses)}


def _filter_key(filters: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((filters or {}).items()))


def _lexical_similarity(*, first: str, second: str) -> float:
    first_tokens = set(_tokens(first))
    second_tokens = set(_tokens(second))
    union = first_tokens | second_tokens
    return len(first_tokens & second_tokens) / len(union) if union else 0.0


def select_candidates(
    candidates: list[RetrievalHit],
    *,
    limit: int,
    preserve_chunk_ids: set[str] | None = None,
) -> list[RetrievalHit]:
    """Greedily balance fused relevance, query coverage, sources, and redundancy."""
    if limit <= 0 or not candidates:
        return []
    return _select_from_ordered(
        ordered=_rank_candidates(candidates),
        limit=limit,
        preserved=preserve_chunk_ids or set(),
    )


def reciprocal_rank_fusion(
    semantic: list[RetrievalHit],
    *positional_sparse: list[RetrievalHit],
    **arguments: Unpack[FusionArguments],
) -> list[RetrievalHit]:
    sparse_ranking = arguments.get("sparse") or positional_sparse[0]
    limit = arguments["limit"]
    rank_constant = arguments.get("rank_constant", 60)
    scores: dict[str, float] = {}
    hits: dict[str, RetrievalHit] = {}
    component_scores: dict[str, dict[str, float]] = {}
    provenance: dict[str, set[str]] = {}
    for kind, ranking in (("semantic", semantic), ("sparse", sparse_ranking)):
        for rank, item in enumerate(ranking, 1):
            hits.setdefault(item.chunk_id, item)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1 / (rank_constant + rank)
            component_scores.setdefault(item.chunk_id, {})[kind] = item.score
            provenance.setdefault(item.chunk_id, set()).update(item.subqueries)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        hits[chunk_id].model_copy(
            update={
                "score": scores[chunk_id],
                "semantic_score": component_scores[chunk_id].get("semantic"),
                "sparse_score": component_scores[chunk_id].get("sparse"),
                "fused_score": scores[chunk_id],
                "subqueries": sorted(provenance[chunk_id]),
            }
        )
        for chunk_id in ordered
    ]


class Retriever:
    def __init__(self, vectorstore: object):
        if not isinstance(vectorstore, RetrievalVectorStore):
            raise TypeError("vectorstore must support retrieval operations")
        self.vectorstore = vectorstore
        self._sparse_indices: dict[tuple[tuple[str, str], ...], _SparseIndex] = {}

    def invalidate_sparse_index(self) -> None:
        """Discard cached BM25 indexes after the underlying corpus changes."""
        self._sparse_indices.clear()

    def _sparse_index(self, filters: dict[str, str] | None) -> _SparseIndex | None:
        key = _filter_key(filters)
        if key in self._sparse_indices:
            return self._sparse_indices[key]
        stored = self.vectorstore.get(include=["documents", "metadatas"], where=_where(filters))
        documents = [str(item) for item in stored.get("documents") or []]
        metadatas = [dict(item or {}) for item in stored.get("metadatas") or []]
        if not documents:
            return None
        index = _SparseIndex(
            documents=documents,
            metadatas=metadatas,
            bm25=BM25Okapi([_tokens(text) for text in documents]),
        )
        self._sparse_indices[key] = index
        return index

    def semantic(
        self,
        query: str,
        *positional_k: int,
        k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalHit]:
        limit = k if k is not None else positional_k[0]
        pairs = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=limit, filter=_where(filters)
        )
        return [
            _hit(document=document, score=score, score_type="semantic") for document, score in pairs
        ]

    def sparse(
        self,
        query: str,
        *positional_k: int,
        k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalHit]:
        limit = k if k is not None else positional_k[0]
        index = self._sparse_index(filters)
        if index is None:
            return []
        scores = index.bm25.get_scores(_tokens(query))
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:limit]
        return [
            _hit(
                document=Document(
                    page_content=index.documents[item_index],
                    metadata=index.metadatas[item_index],
                ),
                score=scores[item_index],
                score_type="sparse",
            )
            for item_index in indices
            if scores[item_index] > 0
        ]

    def search(
        self,
        query: str,
        *positional_options: SearchOptions,
        options: SearchOptions | None = None,
        **arguments: Unpack[SearchArguments],
    ) -> RetrievalBatch:
        selected_options = options or (
            positional_options[0] if positional_options else SearchOptions(**arguments)
        )
        semantic = self.semantic(
            query, k=selected_options.semantic_k, filters=selected_options.filters
        )
        if selected_options.strategy == "semantic":
            selected = select_candidates(semantic, limit=selected_options.selection_limit)
            return RetrievalBatch(
                hits=selected,
                retrieved_count=len(semantic),
                fused_count=len(semantic),
                selected_count=len(selected),
            )
        sparse = self.sparse(query, k=selected_options.sparse_k, filters=selected_options.filters)
        fused = reciprocal_rank_fusion(semantic, sparse=sparse, limit=selected_options.fusion_limit)
        selected = select_candidates(fused, limit=selected_options.selection_limit)
        return RetrievalBatch(
            hits=selected,
            retrieved_count=len(semantic) + len(sparse),
            fused_count=len(fused),
            selected_count=len(selected),
        )
