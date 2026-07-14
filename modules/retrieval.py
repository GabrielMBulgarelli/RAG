"""Semantic and BM25 retrieval with reciprocal-rank fusion."""

import re
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .models import RetrievalHit


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
        stored = self.vectorstore.get(include=["documents", "metadatas"], where=_where(filters))
        documents = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []
        if not documents:
            return []
        bm25 = BM25Okapi([_tokens(text) for text in documents])
        scores = bm25.get_scores(_tokens(query))
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:k]
        return [
            _hit(
                Document(page_content=documents[index], metadata=metadatas[index] or {}),
                scores[index],
                score_type="sparse",
            )
            for index in indices
            if scores[index] > 0
        ]

    def search(
        self,
        query: str,
        *,
        strategy: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalHit]:
        semantic = self.semantic(query, k, filters=filters)
        if strategy == "semantic":
            return semantic
        return reciprocal_rank_fusion(semantic, self.sparse(query, k, filters=filters), limit=k)
