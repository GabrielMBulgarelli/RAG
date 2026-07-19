"""Deterministic context labels and citation validation."""

import re

from .models import AnswerValidation, AnswerViolation, CitationSource, RetrievalHit

LABEL_PATTERN = re.compile(r"\[C(\d+)\]")


def build_cited_context(hits: list[RetrievalHit]) -> tuple[str, list[CitationSource]]:
    sources: list[CitationSource] = []
    blocks: list[str] = []
    for index, hit in enumerate(hits, 1):
        label = f"C{index}"
        excerpt = " ".join(hit.content.split())[:300]
        sources.append(
            CitationSource(
                label=label,
                chunk_id=hit.chunk_id,
                filename=hit.filename,
                page=hit.page,
                excerpt=excerpt,
            )
        )
        blocks.append(f"[{label}] {hit.filename}, page {hit.page}\n{hit.content}")
    return "\n\n".join(blocks), sources


def build_relevant_context(
    hits: list[RetrievalHit], relevant_labels: set[str]
) -> tuple[str, list[CitationSource]]:
    """Build context from relevant hits while preserving labels from the full ranking."""
    sources: list[CitationSource] = []
    blocks: list[str] = []
    for index, hit in enumerate(hits, 1):
        label = f"C{index}"
        if label not in relevant_labels:
            continue
        excerpt = " ".join(hit.content.split())[:300]
        sources.append(
            CitationSource(
                label=label,
                chunk_id=hit.chunk_id,
                filename=hit.filename,
                page=hit.page,
                excerpt=excerpt,
            )
        )
        blocks.append(f"[{label}] {hit.filename}, page {hit.page}\n{hit.content}")
    return "\n\n".join(blocks), sources


def validate_answer(
    answer: str,
    sources: list[CitationSource],
    *,
    known_labels: set[str] | None = None,
    require_citations: bool = True,
) -> AnswerValidation:
    """Validate grounding and return a sanitized answer plus exact cited sources."""
    relevant = {source.label: source for source in sources}
    known = set(relevant) if known_labels is None else known_labels
    used_labels: list[str] = []
    unknown_labels: list[str] = []
    irrelevant_labels: list[str] = []

    def replace(match: re.Match[str]) -> str:
        label = f"C{match.group(1)}"
        if label not in known:
            if label not in unknown_labels:
                unknown_labels.append(label)
            return ""
        if label not in relevant:
            if label not in irrelevant_labels:
                irrelevant_labels.append(label)
            return ""
        if label not in used_labels:
            used_labels.append(label)
        return match.group(0)

    sanitized = LABEL_PATTERN.sub(replace, answer)
    empty_answer = not answer.strip()
    prose = LABEL_PATTERN.sub("", answer)
    citations_only = bool(answer.strip()) and not re.search(r"\w", prose)
    uncited_claims: list[str] = []
    if require_citations and not empty_answer and not citations_only:
        for claim in re.split(r"(?<=[.!?;])(?:\s+|$)|\n+", sanitized):
            claim = claim.strip()
            if re.search(r"\w", claim) and not LABEL_PATTERN.search(claim):
                uncited_claims.append(claim)

    violations: list[AnswerViolation] = []
    if empty_answer:
        violations.append(AnswerViolation.EMPTY_ANSWER)
    if citations_only:
        violations.append(AnswerViolation.CITATIONS_ONLY)
    if unknown_labels:
        violations.append(AnswerViolation.UNKNOWN_LABEL)
    if irrelevant_labels:
        violations.append(AnswerViolation.IRRELEVANT_CITATION)
    if uncited_claims:
        violations.append(AnswerViolation.UNCITED_CLAIM)
    return AnswerValidation(
        sanitized_text=sanitized,
        used_sources=[relevant[label] for label in used_labels],
        violations=violations,
        unknown_labels=unknown_labels,
        irrelevant_labels=irrelevant_labels,
        uncited_claims=uncited_claims,
        empty_answer=empty_answer,
        citations_only=citations_only,
        is_valid=not violations,
    )


def validate_citations(
    answer: str, sources: list[CitationSource]
) -> tuple[str, list[CitationSource]]:
    validation = validate_answer(answer, sources, require_citations=False)
    return validation.sanitized_text, validation.used_sources
