from pathlib import Path

from langchain_core.documents import Document

from modules.citations import build_cited_context, validate_citations
from modules.models import EvidenceStatus, RetrievalHit
from modules.rag_graph import decide_after_grading
from modules.retrieval import reciprocal_rank_fusion
from modules.vector_db import VectorDBManager


def hit(chunk_id: str, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        filename="guide.txt",
        page=1,
        score=score,
    )


def test_chunk_ids_are_stable_and_distinguish_pages(tmp_path: Path) -> None:
    manager = VectorDBManager()
    documents = [
        Document(page_content="same text", metadata={"filename": "a.pdf", "page": 1}),
        Document(page_content="same text", metadata={"filename": "a.pdf", "page": 2}),
    ]

    first = manager.prepare_chunks(documents)
    second = manager.prepare_chunks(documents)

    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]
    assert first[0].metadata["chunk_id"] != first[1].metadata["chunk_id"]


def test_reciprocal_rank_fusion_deduplicates_chunk_ids() -> None:
    fused = reciprocal_rank_fusion([hit("a"), hit("b")], [hit("b"), hit("c")], limit=3)

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert all(item.score > 0 for item in fused)


def test_citations_are_deterministic_and_invalid_labels_are_removed() -> None:
    context, sources = build_cited_context([hit("b"), hit("a")])
    answer, cited_sources = validate_citations("Use [C1] and ignore [C9].", sources)

    assert context.startswith("[C1] guide.txt, page 1")
    assert "[C2]" in context
    assert answer == "Use [C1] and ignore ."
    assert [source.label for source in cited_sources] == ["C1"]


def test_evidence_status_is_not_a_yes_no_string() -> None:
    assert {status.value for status in EvidenceStatus} == {"sufficient", "limited", "insufficient"}


def test_insufficient_evidence_retries_only_once() -> None:
    assert decide_after_grading(EvidenceStatus.INSUFFICIENT, 0) == "retry"
    assert decide_after_grading(EvidenceStatus.INSUFFICIENT, 1) == "abstain"
    assert decide_after_grading(EvidenceStatus.LIMITED, 0) == "answer"
