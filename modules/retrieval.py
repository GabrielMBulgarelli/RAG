"""Semantic and BM25 retrieval with reciprocal-rank fusion."""

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .models import RetrievalBatch, RetrievalHit


@dataclass(frozen=True)
class _SparseIndex:
    documents: list[str]
    metadatas: list[dict[str, Any]]
    bm25: BM25Okapi


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.lower())


def _hit(document: Document, score: float, *, score_type: str) -> RetrievalHit:
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


def _where(filters: dict[str, str] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses = [{key: value} for key, value in sorted(filters.items())]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _filter_key(filters: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((filters or {}).items()))


def _lexical_similarity(first: str, second: str) -> float:
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
    ordered = sorted(
        candidates,
        key=lambda item: (
            -(item.fused_score if item.fused_score is not None else item.score),
            item.chunk_id,
        ),
    )
    relevance = {item.chunk_id: 1.0 - (rank / len(ordered)) for rank, item in enumerate(ordered)}
    all_queries = {query for item in ordered for query in item.subqueries}
    selected: list[RetrievalHit] = []
    remaining = list(ordered)
    preserved = preserve_chunk_ids or set()
    covered_queries: set[str] = set()
    covered_documents: set[str] = set()

    while remaining and len(selected) < limit:
        scored: list[tuple[bool, float, float, str, RetrievalHit]] = []
        for item in remaining:
            new_queries = set(item.subqueries) - covered_queries
            coverage_bonus = 0.2 * len(new_queries) / max(len(all_queries), 1)
            document_key = item.document_id or item.filename
            diversity_bonus = 0.1 if document_key not in covered_documents else 0.0
            redundancy = max(
                (_lexical_similarity(item.content, chosen.content) for chosen in selected),
                default=0.0,
            )
            selection_score = (
                0.7 * relevance[item.chunk_id]
                + coverage_bonus
                + diversity_bonus
                - 0.25 * redundancy
            )
            scored.append(
                (
                    item.chunk_id in preserved,
                    selection_score,
                    relevance[item.chunk_id],
                    item.chunk_id,
                    item,
                )
            )
        _, selection_score, _, _, chosen = min(
            scored, key=lambda value: (-value[0], -value[1], -value[2], value[3])
        )
        chosen = chosen.model_copy(update={"selection_score": selection_score})
        selected.append(chosen)
        remaining = [item for item in remaining if item.chunk_id != chosen.chunk_id]
        covered_queries.update(chosen.subqueries)
        covered_documents.add(chosen.document_id or chosen.filename)
    return selected


def reciprocal_rank_fusion(
    semantic: list[RetrievalHit], sparse: list[RetrievalHit], *, limit: int, rank_constant: int = 60
) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    hits: dict[str, RetrievalHit] = {}
    component_scores: dict[str, dict[str, float]] = {}
    provenance: dict[str, set[str]] = {}
    for kind, ranking in (("semantic", semantic), ("sparse", sparse)):
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
    def __init__(self, vectorstore):
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
        self, query: str, k: int, *, filters: dict[str, str] | None = None
    ) -> list[RetrievalHit]:
        pairs = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=_where(filters)
        )
        return [_hit(document, score, score_type="semantic") for document, score in pairs]

    def sparse(
        self, query: str, k: int, *, filters: dict[str, str] | None = None
    ) -> list[RetrievalHit]:
        index = self._sparse_index(filters)
        if index is None:
            return []
        scores = index.bm25.get_scores(_tokens(query))
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:k]
        return [
            _hit(
                Document(
                    page_content=index.documents[item_index],
                    metadata=index.metadatas[item_index],
                ),
                scores[item_index],
                score_type="sparse",
            )
            for item_index in indices
            if scores[item_index] > 0
        ]

    def search(
        self,
        query: str,
        *,
        strategy: str,
        semantic_k: int,
        sparse_k: int,
        fusion_limit: int,
        selection_limit: int,
        filters: dict[str, str] | None = None,
    ) -> RetrievalBatch:
        semantic = self.semantic(query, semantic_k, filters=filters)
        if strategy == "semantic":
            selected = select_candidates(semantic, limit=selection_limit)
            return RetrievalBatch(
                hits=selected,
                retrieved_count=len(semantic),
                fused_count=len(semantic),
                selected_count=len(selected),
            )
        sparse = self.sparse(query, sparse_k, filters=filters)
        fused = reciprocal_rank_fusion(semantic, sparse, limit=fusion_limit)
        selected = select_candidates(fused, limit=selection_limit)
        return RetrievalBatch(
            hits=selected,
            retrieved_count=len(semantic) + len(sparse),
            fused_count=len(fused),
            selected_count=len(selected),
        )
