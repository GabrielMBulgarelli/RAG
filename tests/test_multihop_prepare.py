import json
from pathlib import Path

import pytest

from modules.config import Settings
from modules.evaluation import EvaluationCase, preflight_multihop
from scripts.prepare_multihop_eval import select_question_rows, stable_document_id


def test_fixed_stratified_selection_is_disjoint_and_reproducible() -> None:
    rows = [
        {"question_type": kind, "query": f"{kind}-{index}"}
        for kind in ("inference_query", "comparison_query", "temporal_query", "null_query")
        for index in range(20)
    ]
    first = select_question_rows(rows, seed=20260711)
    second = select_question_rows(rows, seed=20260711)
    assert first == second
    assert len(first["development"]) == 20
    assert len(first["test"]) == 10
    assert {row["original_index"] for row in first["development"]}.isdisjoint(
        {row["original_index"] for row in first["test"]}
    )
    assert sorted(row["question_type"] for row in first["development"]).count("null_query") == 5


def test_stable_document_id_uses_source_identity() -> None:
    record = {"url": "https://example.test/a", "title": "A", "source": "Example"}
    assert stable_document_id(record) == stable_document_id(record)
    assert stable_document_id(record) != stable_document_id(
        {**record, "url": "https://example.test/b"}
    )


def test_multihop_preflight_resolves_exact_evidence_without_llm(tmp_path: Path) -> None:
    sources = tmp_path / "corpus"
    sources.mkdir()
    source_file = sources / "doc-abc.txt"
    source_file.write_text("Header\nThe exact gold fact appears here.\n", encoding="utf-8")
    settings = Settings(
        sources_dir=sources,
        data_dir=tmp_path / "runtime",
        chroma_dir=tmp_path / "runtime" / "chroma",
        manifest_path=tmp_path / "runtime" / "manifest.json",
        trace_dir=tmp_path / "runtime" / "traces",
        logs_dir=tmp_path / "logs",
    )
    from langchain_core.embeddings import Embeddings

    from modules.vector_db import VectorDBManager

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return [float(len(text)), 1.0]

    manager = VectorDBManager(settings, embeddings=FakeEmbeddings())
    indexed = manager.index_document(source_file)
    assert indexed.success
    source_map = {
        "doc-abc": {
            "benchmark_document_id": "doc-abc",
            "relative_path": "doc-abc.txt",
            "document_id": indexed.document_id,
        }
    }
    map_path = tmp_path / "source_map.json"
    map_path.write_text(json.dumps(source_map), encoding="utf-8")
    map_path.with_name("cases.jsonl").write_text("{}\n", encoding="utf-8")
    cases = [
        EvaluationCase.model_validate(
            {
                "id": "q-1",
                "split": "development",
                "category": "inference_query",
                "question": "Question?",
                "answerable": True,
                "expected_route": "complex_search",
                "expected_strategy": "hybrid",
                "expected_retry": False,
                "expected_conflict": False,
                "expected_answer": "answer",
                "gold_evidence": [
                    {
                        "benchmark_document_id": "doc-abc",
                        "source": "Example",
                        "title": "A",
                        "url": "https://example.test/a",
                        "evidence_text": "The exact gold fact appears here.",
                    }
                ],
            }
        )
    ]
    resolved = preflight_multihop(cases, map_path, manager, check_models=False)
    assert resolved[0].relevant_chunk_ids
    assert resolved[0].relevant_document_ids == [indexed.document_id]


def test_multihop_preflight_fails_before_model_check_on_unresolved_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def model_check() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("modules.evaluation._require_ollama", model_check)
    with pytest.raises(RuntimeError, match="benchmark files"):
        preflight_multihop([], tmp_path / "missing.json", None, check_models=True)
    assert called is False
