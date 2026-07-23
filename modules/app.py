"""Backward-compatible entry point for the refactored UI application."""

from __future__ import annotations

import gradio as gr

from modules.config import config
from modules.ui.application import *  # noqa: F403
from modules.ui.application import RAGApplication
from modules.ui.shell import build_application

app = RAGApplication()


def create_interface() -> gr.Blocks:
    """Build the routed dashboard using the shared application controller."""
    return build_application(app)


def main() -> int:
    create_interface().launch(
        server_name=config.gradio_host,
        server_port=config.gradio_port,
        share=config.gradio_share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
