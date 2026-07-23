"""Automatic ingestion and Inspector document-management bindings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import gradio as gr
from gradio.blocks import BlockContext
from gradio.components import Component

from modules.ui.events import (
    ComponentUpdate,
    knowledge_inventory_update,
    knowledge_selection_reset_updates,
)
from modules.ui.pages.ask_callbacks import render_corpus_context
from modules.ui.pages.document_components import (
    IndexedDocumentComponents,
    SidebarDocumentComponents,
)
from modules.ui.presenters import render_indexing_errors

if TYPE_CHECKING:
    from modules.ui.application import RAGApplication

SelectionUpdates = tuple[str, ComponentUpdate, ComponentUpdate, str, ComponentUpdate]
RefreshUpdates = tuple[
    ComponentUpdate,
    ComponentUpdate,
    str,
    ComponentUpdate,
    ComponentUpdate,
    str,
    ComponentUpdate,
]
DeletionReviewUpdates = tuple[str, ComponentUpdate]
DeletionUpdates = tuple[ComponentUpdate, str, ComponentUpdate]

_INDEXING_CONCURRENCY_ID = "document-indexing"


@dataclass(frozen=True)
class DocumentBindingTargets:
    indexed_documents: IndexedDocumentComponents
    sidebar: SidebarDocumentComponents
    corpus_context: gr.HTML
    sidebar_status: gr.HTML
    refresh_sidebar: Callable[[], str]


def indexing_errors_update(rows: Sequence[Sequence[object]]) -> ComponentUpdate:
    """Hide an empty error section and reveal actionable indexing failures."""
    return cast(
        ComponentUpdate,
        gr.update(value=render_indexing_errors(rows), visible=bool(rows)),
    )


def index_uploaded_documents(  # lanorme: ignore[KWARG-001] -- Gradio injects progress
    application: RAGApplication,
    files: list[str] | None,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[ComponentUpdate, ComponentUpdate]:
    """Index a non-empty selection immediately and clear the file picker."""
    if not files:
        skipped = cast(ComponentUpdate, gr.skip())
        return skipped, skipped
    status = cast(ComponentUpdate, application.index_selected_action_ui(files, progress))
    return cast(ComponentUpdate, gr.update(value=None)), status


def refresh_documents(application: RAGApplication) -> RefreshUpdates:
    """Refresh inventory and errors while clearing any stale row selection."""
    rows = application.document_samples()
    reset = knowledge_selection_reset_updates(has_documents=bool(rows))
    return (
        knowledge_inventory_update(rows),
        indexing_errors_update(application.current_error_rows()),
        *reset,
    )


def select_document(  # lanorme: ignore[KWARG-001] -- Gradio supplies event data
    application: RAGApplication,
    event: gr.SelectData,
) -> SelectionUpdates:
    return cast(SelectionUpdates, application.select_document(None, event))


def prepare_deletion(
    *,
    application: RAGApplication,
    document_id: str | None,
) -> DeletionReviewUpdates:
    return cast(DeletionReviewUpdates, application.prepare_deletion(document_id))


def cancel_deletion(application: RAGApplication) -> DeletionReviewUpdates:
    return cast(DeletionReviewUpdates, application.cancel_deletion())


def delete_document(
    *,
    application: RAGApplication,
    document_id: str | None,
) -> DeletionUpdates:
    return cast(DeletionUpdates, application.delete_selected_action_ui(document_id))


def _document_outputs(
    components: IndexedDocumentComponents,
) -> tuple[Component | BlockContext, ...]:
    return (
        components.inventory,
        components.indexing_errors,
        components.selected_document_id,
        components.selected_document,
        components.delete_selected,
        components.deletion_text,
        components.deletion_confirmation,
    )


def _bind_inventory_controls(
    *,
    application: RAGApplication,
    components: IndexedDocumentComponents,
) -> None:
    def handle_selection(event: gr.SelectData) -> SelectionUpdates:
        return select_document(application, event)

    components.inventory.select(
        fn=handle_selection,
        outputs=[
            components.selected_document_id,
            components.selected_document,
            components.delete_selected,
            components.deletion_text,
            components.deletion_confirmation,
        ],
        show_progress="hidden",
    )


def _bind_inventory(
    *,
    application: RAGApplication,
    targets: DocumentBindingTargets,
) -> None:
    components = targets.indexed_documents
    outputs = _document_outputs(components)
    refresh = partial(refresh_documents, application)
    gr.on(triggers=None, fn=refresh, outputs=outputs, show_progress="hidden")
    _bind_inventory_controls(application=application, components=components)
    components.delete_selected.click(
        fn=lambda document_id: prepare_deletion(application=application, document_id=document_id),
        inputs=components.selected_document_id,
        outputs=[components.deletion_text, components.deletion_confirmation],
    )
    components.cancel_delete.click(
        fn=partial(cancel_deletion, application),
        outputs=[components.deletion_text, components.deletion_confirmation],
    )
    delete_event = components.confirm_delete.click(
        fn=lambda document_id: delete_document(application=application, document_id=document_id),
        inputs=components.selected_document_id,
        outputs=[
            targets.sidebar.status,
            components.selected_document_id,
            components.deletion_confirmation,
        ],
    )
    delete_event.then(
        fn=refresh,
        outputs=outputs,
        show_progress="hidden",
    ).then(
        fn=partial(render_corpus_context, application),
        outputs=targets.corpus_context,
        show_progress="hidden",
    ).then(
        fn=targets.refresh_sidebar,
        outputs=targets.sidebar_status,
        show_progress="hidden",
    )


def bind_document_management(  # lanorme: ignore[PARAM-001] -- shared shell binding
    *,
    application: RAGApplication,
    sidebar: SidebarDocumentComponents,
    sidebar_status: gr.HTML,
    refresh_sidebar: Callable[[], str],
    indexed_documents: IndexedDocumentComponents | None = None,
    corpus_context: gr.HTML | None = None,
) -> None:
    """Bind shared automatic ingestion and optional Ask inventory refreshes."""
    upload_event = sidebar.upload.change(
        fn=partial(index_uploaded_documents, application),
        inputs=sidebar.upload,
        outputs=[sidebar.upload, sidebar.status],
        concurrency_limit=1,
        concurrency_id=_INDEXING_CONCURRENCY_ID,
        api_name="index_uploaded_documents",
    )
    refresh_event = upload_event
    if indexed_documents is not None and corpus_context is not None:
        _bind_inventory(
            application=application,
            targets=DocumentBindingTargets(
                indexed_documents=indexed_documents,
                sidebar=sidebar,
                corpus_context=corpus_context,
                sidebar_status=sidebar_status,
                refresh_sidebar=refresh_sidebar,
            ),
        )
        refresh_event = refresh_event.then(
            fn=partial(refresh_documents, application),
            outputs=_document_outputs(indexed_documents),
            show_progress="hidden",
        ).then(
            fn=partial(render_corpus_context, application),
            outputs=corpus_context,
            show_progress="hidden",
        )
    refresh_event.then(fn=refresh_sidebar, outputs=sidebar_status, show_progress="hidden")
