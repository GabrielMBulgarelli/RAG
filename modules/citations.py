"""Deterministic context labels and citation validation."""

import re

from .models import CitationSource, RetrievalHit

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


def validate_citations(
    answer: str, sources: list[CitationSource]
) -> tuple[str, list[CitationSource]]:
    valid = {source.label: source for source in sources}
    cited: list[str] = []

    def replace(match: re.Match[str]) -> str:
        label = f"C{match.group(1)}"
        if label not in valid:
            return ""
        if label not in cited:
            cited.append(label)
        return match.group(0)

    cleaned = LABEL_PATTERN.sub(replace, answer)
    return cleaned, [valid[label] for label in cited]
