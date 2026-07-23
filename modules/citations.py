"""Deterministic context labels and citation validation."""

import re
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict, Unpack

from .models import AnswerValidation, AnswerViolation, CitationSource, RetrievalHit

LABEL_PATTERN = re.compile(r"\[C(\d+)\]")


class ValidationArguments(TypedDict):
    sources: NotRequired[list[CitationSource]]
    known_labels: NotRequired[set[str] | None]
    require_citations: NotRequired[bool]


@dataclass
class _CitationSanitizer:
    relevant: dict[str, CitationSource]
    known: set[str]
    used: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    irrelevant: list[str] = field(default_factory=list)

    def replace(self, match: re.Match[str]) -> str:
        label = f"C{match.group(1)}"
        if label not in self.known:
            if label not in self.unknown:
                self.unknown.append(label)
            return ""
        if label not in self.relevant:
            if label not in self.irrelevant:
                self.irrelevant.append(label)
            return ""
        if label not in self.used:
            self.used.append(label)
        return match.group(0)


def _sanitize_labels(
    *, answer: str, relevant: dict[str, CitationSource], known: set[str]
) -> tuple[str, list[str], list[str], list[str]]:
    sanitizer = _CitationSanitizer(relevant=relevant, known=known)
    sanitized = LABEL_PATTERN.sub(sanitizer.replace, answer)
    return sanitized, sanitizer.used, sanitizer.unknown, sanitizer.irrelevant


def _find_uncited_claims(*, sanitized: str, required: bool) -> list[str]:
    if not required:
        return []
    claims = (claim.strip() for claim in re.split(r"(?<=[.!?;])(?:\s+|$)|\n+", sanitized))
    return [
        claim for claim in claims if re.search(r"\w", claim) and not LABEL_PATTERN.search(claim)
    ]


def _answer_violations(
    checks: tuple[tuple[bool, AnswerViolation], ...],
) -> list[AnswerViolation]:
    return [violation for applies, violation in checks if applies]


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
    hits: list[RetrievalHit],
    *positional_labels: set[str],
    relevant_labels: set[str] | None = None,
) -> tuple[str, list[CitationSource]]:
    """Build context from relevant hits while preserving labels from the full ranking."""
    selected_labels = relevant_labels if relevant_labels is not None else positional_labels[0]
    sources: list[CitationSource] = []
    blocks: list[str] = []
    for index, hit in enumerate(hits, 1):
        label = f"C{index}"
        if label not in selected_labels:
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
    *positional_sources: list[CitationSource],
    **arguments: Unpack[ValidationArguments],
) -> AnswerValidation:
    """Validate grounding and return a sanitized answer plus exact cited sources."""
    selected_sources = arguments.get("sources") or positional_sources[0]
    known_labels = arguments.get("known_labels")
    require_citations = arguments.get("require_citations", True)
    relevant = {source.label: source for source in selected_sources}
    known = set(relevant) if known_labels is None else known_labels
    sanitized, used_labels, unknown_labels, irrelevant_labels = _sanitize_labels(
        answer=answer, relevant=relevant, known=known
    )
    empty_answer = not answer.strip()
    prose = LABEL_PATTERN.sub("", answer)
    citations_only = bool(answer.strip()) and not re.search(r"\w", prose)
    uncited_claims = _find_uncited_claims(
        sanitized=sanitized,
        required=require_citations and not empty_answer and not citations_only,
    )
    violations = _answer_violations(
        (
            (empty_answer, AnswerViolation.EMPTY_ANSWER),
            (citations_only, AnswerViolation.CITATIONS_ONLY),
            (bool(unknown_labels), AnswerViolation.UNKNOWN_LABEL),
            (bool(irrelevant_labels), AnswerViolation.IRRELEVANT_CITATION),
            (bool(uncited_claims), AnswerViolation.UNCITED_CLAIM),
        )
    )
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


def retain_cited_claims(
    answer: str,
    sources: list[CitationSource],
    *,
    known_labels: set[str],
) -> AnswerValidation:
    """Drop uncited prose while preserving independently cited claims."""
    claims = re.split(r"(?<=[.!?;])(?:\s+|$)|\n+", answer)
    cited_only = " ".join(claim.strip() for claim in claims if LABEL_PATTERN.search(claim))
    return validate_answer(
        cited_only,
        sources,
        known_labels=known_labels,
        require_citations=True,
    )


def validate_citations(
    answer: str,
    *positional_sources: list[CitationSource],
    sources: list[CitationSource] | None = None,
) -> tuple[str, list[CitationSource]]:
    selected_sources = sources if sources is not None else positional_sources[0]
    validation = validate_answer(answer, sources=selected_sources, require_citations=False)
    return validation.sanitized_text, validation.used_sources
