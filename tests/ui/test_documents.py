from typing import Any, cast

import gradio as gr

from modules.ui.contracts import CorpusSnapshot
from modules.ui.pages.documents import (
    cancel_deletion,
    delete_document,
    index_uploaded_documents,
    indexing_errors_update,
    prepare_deletion,
    refresh_documents,
    select_document,
)
from modules.ui.shell import build_application


class DocumentApplication:
    rag_graph = None

    def __init__(self) -> None:
        self.index_calls: list[list[str] | None] = []
        self.rows = [["manual.txt", 2, 7, "Indexed"]]
        self.errors: list[list[str]] = []

    def index_selected_action_ui(
        self,
        files: list[str] | None,
        _progress: gr.Progress,
    ) -> dict[str, object]:
        self.index_calls.append(files)
        return {"value": "Indexed documents", "visible": True}

    def document_samples(self) -> list[list[object]]:
        return self.rows

    def current_error_rows(self) -> list[list[str]]:
        return self.errors

    def corpus_snapshot(self) -> CorpusSnapshot:
        return CorpusSnapshot(1, 2, 7, "ready")

    def select_document(
        self,
        _query: None,
        _event: gr.SelectData,
    ) -> tuple[str, dict[str, object], dict[str, object], str, dict[str, object]]:
        return (
            "manual.txt",
            {"value": "manual.txt", "visible": True},
            {"visible": True, "interactive": True},
            "",
            {"visible": False},
        )

    def prepare_deletion(self, document_id: str | None) -> tuple[str, dict[str, object]]:
        return f"Delete {document_id}?", {"visible": True}

    def cancel_deletion(self) -> tuple[str, dict[str, object]]:
        return "", {"visible": False}

    def delete_selected_action_ui(
        self, document_id: str | None
    ) -> tuple[dict[str, object], str, dict[str, object]]:
        return {"value": f"Deleted {document_id}"}, "", {"visible": False}


def test_indexed_documents_are_in_the_ask_inspector_without_search_or_maintenance() -> None:
    # Arrange
    config = build_application(cast(Any, DocumentApplication())).get_config_file()
    components = config["components"]
    ids = {component["props"].get("elem_id") for component in components}
    values = {
        component["props"].get("value")
        for component in components
        if isinstance(component["props"].get("value"), str)
    }
    inventory = next(
        component
        for component in components
        if component["props"].get("elem_id") == "inspector-document-inventory"
    )

    # Assert the Inspector exposes only the supported document controls.
    assert inventory["type"] == "dataset"
    assert inventory["props"]["headers"] == ["Document", "Pages", "Chunks", "Status"]
    assert {
        "inspector-document-inventory",
        "inspector-selected-document",
        "inspector-delete-confirmation",
        "inspector-indexing-errors",
    } <= ids
    assert {"knowledge-filter", "knowledge-maintenance"}.isdisjoint(ids)
    assert {
        "Reindex Changed Documents",
        "Check Index Consistency",
        "Rebuild Index",
    }.isdisjoint(values)


def test_nonempty_upload_indexes_automatically_and_resets_picker() -> None:
    # Arrange
    application = DocumentApplication()

    # Act
    picker, status = index_uploaded_documents(
        cast(Any, application),
        ["manual.txt", "guide.pdf"],
    )

    assert application.index_calls == [["manual.txt", "guide.pdf"]]
    assert picker.get("value") is None
    assert status.get("value") == "Indexed documents"
    assert status.get("visible") is True


def test_empty_or_cancelled_upload_does_nothing() -> None:
    # Arrange
    application = DocumentApplication()

    # Act
    picker, status = index_uploaded_documents(cast(Any, application), [])

    # Assert cancellation leaves both outputs and application state untouched.
    assert application.index_calls == []
    assert picker == gr.skip()
    assert status == gr.skip()


def test_upload_handlers_share_one_serialized_concurrency_group() -> None:
    # Arrange
    interface = build_application(cast(Any, DocumentApplication()))

    # Act
    upload_functions = [
        function
        for function in interface.fns.values()
        if str(function.api_name).startswith("index_uploaded_documents")
    ]

    # Assert the single route uses the serialized indexing queue.
    assert len(upload_functions) == 1
    assert all(function.concurrency_limit == 1 for function in upload_functions)
    assert len({function.concurrency_id for function in upload_functions}) == 1


def test_inventory_refresh_exposes_partial_failures_and_resets_stale_selection() -> None:
    # Arrange
    application = DocumentApplication()
    application.errors = [["bad.pdf", "Index", "Parse error", "Invalid content"]]

    # Act
    inventory, errors, selected_id, selected, delete, confirmation_text, confirmation = (
        refresh_documents(cast(Any, application))
    )

    # Assert partial failures remain visible while stale selection state clears.
    assert inventory.get("samples") == application.rows
    assert errors.get("visible") is True
    assert "bad.pdf" in str(errors.get("value"))
    assert selected_id == ""
    assert selected.get("visible") is False
    assert delete.get("interactive") is False
    assert confirmation_text == ""
    assert confirmation.get("visible") is False


def test_document_selection_cancellation_and_confirmed_deletion_delegate_to_application() -> None:
    # Arrange
    application = DocumentApplication()
    event = cast(gr.SelectData, object())

    # Act
    selection = select_document(cast(Any, application), event)
    confirmation = prepare_deletion(application=cast(Any, application), document_id=selection[0])
    cancellation = cancel_deletion(cast(Any, application))
    deletion = delete_document(application=cast(Any, application), document_id=selection[0])

    # Assert selection and both confirmation outcomes retain backend semantics.
    assert selection[0] == "manual.txt"
    assert selection[1].get("visible") is True
    assert selection[2].get("interactive") is True
    assert confirmation[1].get("visible") is True
    assert cancellation == ("", {"visible": False})
    assert "Deleted manual.txt" in str(deletion[0].get("value"))


def test_indexing_errors_are_hidden_when_empty_and_shown_when_present() -> None:
    # Act
    empty = indexing_errors_update([])
    populated = indexing_errors_update([["manual.txt", "Index", "Parse error", "Invalid content"]])

    # Assert error details appear only when actionable rows exist.
    assert empty.get("visible") is False
    assert populated.get("visible") is True
    assert "manual.txt" in str(populated.get("value"))
