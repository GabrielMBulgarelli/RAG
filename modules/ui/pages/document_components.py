"""Reusable document controls for the shared sidebar and Ask Inspector."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from modules.ui.presenters import render_status

DOCUMENT_HEADERS = ["Document", "Pages", "Chunks", "Status"]


@dataclass(frozen=True)
class SidebarDocumentComponents:
    upload: gr.File
    status: gr.HTML


@dataclass(frozen=True)
class IndexedDocumentComponents:
    selected_document_id: gr.State
    inventory: gr.Dataset
    selected_document: gr.HTML
    delete_selected: gr.Button
    deletion_text: gr.HTML
    deletion_confirmation: gr.Group
    cancel_delete: gr.Button
    confirm_delete: gr.Button
    indexing_errors: gr.HTML


def _build_deletion_components() -> tuple[gr.Button, gr.HTML, gr.Group, gr.Button, gr.Button]:
    delete_selected = gr.Button(
        "Delete Document",
        visible=False,
        interactive=False,
        elem_classes=["document-delete-review"],
    )
    with gr.Group(
        visible=False,
        elem_id="inspector-delete-confirmation",
        elem_classes=["document-delete-confirmation"],
    ) as deletion_confirmation:
        deletion_text = gr.HTML()
        gr.Markdown(
            "This removes the document and its indexed chunks. This action cannot be undone."
        )
        with gr.Row():
            cancel_delete = gr.Button("Cancel", variant="secondary")
            confirm_delete = gr.Button("Delete Document", variant="stop")
    return (
        delete_selected,
        deletion_text,
        deletion_confirmation,
        cancel_delete,
        confirm_delete,
    )


def build_sidebar_document_components() -> SidebarDocumentComponents:
    """Create the automatic ingestion surface shared by every route."""
    with gr.Group(elem_id="sidebar-document-ingestion"):
        upload = gr.File(
            label="Drop files here or click to upload",
            show_label=False,
            file_count="multiple",
            file_types=[".pdf", ".txt"],
            type="filepath",
            elem_id="sidebar-document-upload",
        )
        status = gr.HTML(
            render_status("info", "Ready", "Choose PDF or TXT files to index."),
            visible=False,
            elem_id="sidebar-document-status",
        )
    return SidebarDocumentComponents(upload, status)


def build_indexed_document_components() -> IndexedDocumentComponents:
    """Create the interactive document inventory used by the Ask Inspector."""
    selected_document_id = gr.State("")
    inventory = gr.Dataset(
        components=["textbox", "number", "number", "textbox"],
        samples=[],
        headers=DOCUMENT_HEADERS,
        label="Indexed documents",
        type="values",
        layout="table",
        samples_per_page=25,
        elem_id="inspector-document-inventory",
    )
    selected_document = gr.HTML(
        "",
        visible=False,
        elem_id="inspector-selected-document",
    )
    (
        delete_selected,
        deletion_text,
        deletion_confirmation,
        cancel_delete,
        confirm_delete,
    ) = _build_deletion_components()
    indexing_errors = gr.HTML(
        "",
        visible=False,
        elem_id="inspector-indexing-errors",
        elem_classes=["overflow-region"],
    )
    return IndexedDocumentComponents(
        selected_document_id,
        inventory,
        selected_document,
        delete_selected,
        deletion_text,
        deletion_confirmation,
        cancel_delete,
        confirm_delete,
        indexing_errors,
    )
