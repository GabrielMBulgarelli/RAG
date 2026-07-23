"""Adapters that translate controller results into Gradio component updates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict, cast

import gradio as gr

from modules.ui.contracts import EvaluationPageSnapshot, RuntimeSnapshot, SystemPageSnapshot


class ComponentUpdate(TypedDict):
    """Typed subset of Gradio update fields used by the routed UI."""

    __type__: NotRequired[str]
    interactive: NotRequired[bool]
    placeholder: NotRequired[str]
    samples: NotRequired[list[list[object]]]
    value: NotRequired[object]
    visible: NotRequired[bool]


def component_update(**values: Any) -> dict[str, Any]:
    """Create a Gradio update from presentation-layer values."""
    return gr.update(**values)


def ask_control_updates(
    runtime: RuntimeSnapshot,
) -> tuple[ComponentUpdate, ComponentUpdate, ComponentUpdate]:
    """Translate runtime readiness into composer, submit, and load-button updates."""
    placeholder = (
        "Ask a question about the indexed documents"
        if runtime.chat_enabled
        else "Load AI models before asking about your documents"
    )
    composer = cast(
        ComponentUpdate,
        component_update(
            interactive=runtime.chat_enabled,
            placeholder=placeholder,
        ),
    )
    submit = cast(
        ComponentUpdate,
        component_update(interactive=runtime.chat_enabled),
    )
    load_models = cast(
        ComponentUpdate,
        component_update(
            visible=runtime.can_load_models,
            interactive=runtime.can_load_models,
        ),
    )
    return composer, submit, load_models


def knowledge_selection_reset_updates(
    *,
    has_documents: bool,
) -> tuple[str, ComponentUpdate, ComponentUpdate, str, ComponentUpdate]:
    """Clear the internal selection and close the destructive confirmation flow."""
    selected = cast(
        ComponentUpdate,
        component_update(value="", visible=False),
    )
    delete = cast(
        ComponentUpdate,
        component_update(visible=False, interactive=False),
    )
    confirmation = cast(
        ComponentUpdate,
        component_update(visible=False),
    )
    return "", selected, delete, "", confirmation


def knowledge_inventory_update(rows: Sequence[Sequence[object]]) -> ComponentUpdate:
    """Replace the public inventory samples without adding internal identifiers."""
    return cast(ComponentUpdate, component_update(samples=[list(row) for row in rows]))


def evaluation_control_update(snapshot: EvaluationPageSnapshot) -> ComponentUpdate:
    """Set the availability of the primary evaluation action."""
    return cast(
        ComponentUpdate,
        component_update(
            interactive=snapshot.state != "blocked",
        ),
    )


def system_control_update(snapshot: SystemPageSnapshot) -> ComponentUpdate:
    """Expose explicit model initialization only when system checks allow it."""
    return cast(
        ComponentUpdate,
        component_update(
            visible=snapshot.can_load_models,
            interactive=snapshot.can_load_models,
        ),
    )
