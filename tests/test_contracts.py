"""Characterization tests for shared dependency contracts."""

from pathlib import Path

from modules.contracts import ApplicationVectorStore, RetrievalVectorStore


class _RetrievalStore:
    def get(
        self,
        *,
        include: list[str],
        where: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del include, where
        return {"documents": [], "metadatas": []}

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        *,
        k: int,
        filter: dict[str, object] | None = None,
    ) -> list[tuple[object, float]]:
        del query, k, filter
        return []


class _SourceSettings:
    sources_dir = Path("sources")


class _ApplicationStore:
    settings = _SourceSettings()

    def setup(self) -> _RetrievalStore:
        return _RetrievalStore()

    def document_names(self) -> list[str]:
        return []

    def manifest(self) -> object:
        raise NotImplementedError

    def save_uploads(self, files: list[object]) -> list[Path]:
        del files
        return []

    def index_document(self, file_path: Path) -> object:
        del file_path
        raise NotImplementedError

    def has_deleted_document(self, filename: str) -> bool:
        del filename
        return False

    delete_document = has_deleted_document

    def rebuild(self) -> int:
        return 0

    def reconcile_index(self) -> object:
        raise NotImplementedError

    def chunk_count(self) -> int:
        return 0


def test_retrieval_store_contract_is_runtime_checkable() -> None:
    assert isinstance(_RetrievalStore(), RetrievalVectorStore)


def test_application_store_contract_is_runtime_checkable() -> None:
    assert isinstance(_ApplicationStore(), ApplicationVectorStore)
