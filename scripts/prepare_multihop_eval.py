#!/usr/bin/env python3
"""Prepare a fixed, reproducible MultiHop-RAG evaluation subset and corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Required, TypedDict, cast

from datasets import load_dataset
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluation import (  # noqa: E402
    MULTIHOP_ROOT,
    BenchmarkEvidence,
    EvaluationCase,
    load_cases,
    multihop_settings,
    preflight_multihop,
)
from modules.vector_db import VectorDBManager  # noqa: E402

DATASET_ID = "yixuantt/MultiHopRAG"
SEED = 20260711
TYPE_ORDER = ("inference_query", "comparison_query", "temporal_query", "null_query")
DEVELOPMENT_QUOTA = {query_type: 5 for query_type in TYPE_ORDER}
TEST_QUOTA = {
    "inference_query": 3,
    "comparison_query": 3,
    "temporal_query": 2,
    "null_query": 2,
}


class CorpusRow(TypedDict, total=False):
    url: str
    source: str
    title: str
    author: str
    published_at: str
    category: str
    body: str


class EvidenceRow(TypedDict, total=False):
    url: str
    source: str
    title: str
    fact: str
    author: str
    category: str
    published_at: str


class QuestionRow(TypedDict, total=False):
    question_type: Required[str]
    query: Required[str]
    evidence_list: list[EvidenceRow]
    answer: str


class SelectedQuestionRow(QuestionRow):
    original_index: Required[int]


class SelectedQuestionRows(TypedDict):
    development: list[SelectedQuestionRow]
    test: list[SelectedQuestionRow]


class SourceMapEntry(TypedDict):
    benchmark_document_id: str
    original_row: int
    relative_path: str
    document_id: str
    source: str
    title: str
    author: str
    published_at: str
    url: str
    category: str


def _corpus_row(record: Mapping[str, object]) -> CorpusRow:
    return cast(CorpusRow, dict(record))


def _question_row(record: Mapping[str, object]) -> QuestionRow:
    return cast(QuestionRow, dict(record))


def stable_document_id(record: CorpusRow | Mapping[str, object]) -> str:
    identity = str(record.get("url") or "").strip()
    if not identity:
        identity = json.dumps(
            {key: record.get(key, "") for key in ("source", "title", "author", "published_at")},
            sort_keys=True,
            ensure_ascii=False,
        )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_question_rows(
    rows: Sequence[QuestionRow | Mapping[str, object]], *, seed: int = SEED
) -> SelectedQuestionRows:
    grouped: dict[str, list[SelectedQuestionRow]] = defaultdict(list)
    for index, raw_row in enumerate(rows):
        row = _question_row(raw_row)
        query_type = str(row["question_type"])
        if query_type in TYPE_ORDER:
            grouped[query_type].append(cast(SelectedQuestionRow, {**row, "original_index": index}))
    rng = random.Random(seed)
    selected = SelectedQuestionRows(development=[], test=[])
    for query_type in TYPE_ORDER:
        candidates = grouped[query_type]
        rng.shuffle(candidates)
        needed = DEVELOPMENT_QUOTA[query_type] + TEST_QUOTA[query_type]
        if len(candidates) < needed:
            raise ValueError(f"Not enough {query_type} rows: need {needed}")
        selected["development"].extend(candidates[: DEVELOPMENT_QUOTA[query_type]])
        selected["test"].extend(candidates[DEVELOPMENT_QUOTA[query_type] : needed])
    return selected


def _write_corpus(records: list[CorpusRow], *, reset: bool) -> dict[str, SourceMapEntry]:
    corpus_dir = MULTIHOP_ROOT / "corpus"
    if reset and corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    settings = multihop_settings()
    manager = VectorDBManager(settings)
    source_map: dict[str, SourceMapEntry] = {}
    for index, record in enumerate(records):
        benchmark_id = stable_document_id(record)
        relative_path = f"{index:04d}-{benchmark_id[:12]}.txt"
        path = corpus_dir / relative_path
        header = [
            f"Title: {record.get('title', '')}",
            f"Source: {record.get('source', '')}",
            f"Author: {record.get('author', '')}",
            f"Published: {record.get('published_at', '')}",
            f"URL: {record.get('url', '')}",
            f"Category: {record.get('category', '')}",
        ]
        path.write_text("\n".join(header) + "\n\n" + str(record.get("body", "")), encoding="utf-8")
        source_map[benchmark_id] = {
            "benchmark_document_id": benchmark_id,
            "original_row": index,
            "relative_path": relative_path,
            "document_id": manager.document_id(path),
            **{
                key: str(record.get(key, ""))
                for key in ("source", "title", "author", "published_at", "url", "category")
            },
        }
    return source_map


def _resolve_evidence_document(
    evidence: EvidenceRow, *, records_by_url: dict[str, CorpusRow]
) -> str:
    url = str(evidence.get("url", ""))
    record = records_by_url.get(url)
    if record is None:
        raise ValueError(f"Evidence URL is absent from corpus: {url}")
    return stable_document_id(record)


def _load_rows() -> tuple[list[CorpusRow], list[QuestionRow]]:
    corpus_rows = [_corpus_row(row) for row in load_dataset(DATASET_ID, "corpus", split="train")]
    question_rows = [
        _question_row(row) for row in load_dataset(DATASET_ID, "MultiHopRAG", split="train")
    ]
    return corpus_rows, question_rows


def _evidence_items(
    row: SelectedQuestionRow, *, records_by_url: dict[str, CorpusRow]
) -> list[BenchmarkEvidence]:
    return [
        BenchmarkEvidence(
            benchmark_document_id=_resolve_evidence_document(
                evidence, records_by_url=records_by_url
            ),
            source=str(evidence.get("source", "")),
            title=str(evidence.get("title", "")),
            url=str(evidence.get("url", "")),
            evidence_text=str(evidence.get("fact", "")),
            author=str(evidence.get("author", "")),
            category=str(evidence.get("category", "")),
            published_at=str(evidence.get("published_at", "")),
        )
        for evidence in row.get("evidence_list", [])
    ]


def _evaluation_case(
    row: SelectedQuestionRow,
    *,
    split: str,
    records_by_url: dict[str, CorpusRow],
) -> EvaluationCase:
    query_type = str(row["question_type"])
    return EvaluationCase(
        id=f"multihop-{row['original_index']:04d}",
        split=split,
        category=query_type,
        question=str(row["query"]),
        answerable=query_type != "null_query",
        gold_evidence=_evidence_items(row, records_by_url=records_by_url),
        expected_answer=str(row.get("answer", "")),
        expected_route="complex_search",
        expected_strategy="hybrid",
        expected_retry=query_type == "null_query",
        expected_conflict=False,
    )


def _build_cases(
    selected: SelectedQuestionRows, *, records_by_url: dict[str, CorpusRow]
) -> list[EvaluationCase]:
    return [
        _evaluation_case(row, split=split, records_by_url=records_by_url)
        for split in ("development", "test")
        for row in selected[split]
    ]


def _write_outputs(
    *, cases: list[EvaluationCase], source_map: dict[str, SourceMapEntry], corpus_count: int
) -> Path:
    MULTIHOP_ROOT.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(DATASET_ID)
    map_payload = {
        "dataset": {
            "id": DATASET_ID,
            "revision": info.sha,
            "license": "ODC-By-1.0",
            "seed": SEED,
            "corpus_documents": corpus_count,
        },
        "documents": source_map,
    }
    (MULTIHOP_ROOT / "source_map.json").write_text(
        json.dumps(map_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cases_path = MULTIHOP_ROOT / "cases.jsonl"
    cases_path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    return cases_path


def _index_corpus(*, cases_path: Path, reset: bool) -> int:
    manager = VectorDBManager(multihop_settings())
    if reset:
        shutil.rmtree(manager.settings.chroma_dir, ignore_errors=True)
        manager.settings.manifest_path.unlink(missing_ok=True)
    manager.index_documents()
    checked = preflight_multihop(load_cases(cases_path), manager=manager, check_models=False)
    return sum(len(case.gold_evidence) for case in checked)


def prepare(*, index: bool = False, reset: bool = False) -> tuple[Path, int]:
    corpus_rows, question_rows = _load_rows()
    if len(corpus_rows) != 609:
        raise RuntimeError(f"Expected 609 corpus documents, received {len(corpus_rows)}")
    source_map = _write_corpus(corpus_rows, reset=reset)
    records_by_url = {str(row.get("url", "")): row for row in corpus_rows}
    selected = select_question_rows(question_rows)
    cases = _build_cases(selected, records_by_url=records_by_url)
    cases_path = _write_outputs(cases=cases, source_map=source_map, corpus_count=len(corpus_rows))
    resolved = _index_corpus(cases_path=cases_path, reset=reset) if index else 0
    return cases_path, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", action="store_true", help="Index with the existing ingestion system"
    )
    parser.add_argument("--reset", action="store_true", help="Recreate corpus and evaluation index")
    args = parser.parse_args()
    cases_path, resolved = prepare(index=args.index, reset=args.reset)
    print(f"Prepared {cases_path}")
    if args.index:
        print(f"Resolved {resolved} gold evidence records against indexed chunks")


if __name__ == "__main__":
    main()
