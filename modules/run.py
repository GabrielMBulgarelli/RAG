"""Package-safe application launcher with actionable prerequisite checks."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from typing import cast
from urllib.error import URLError
from urllib.request import urlopen

from .config import Settings, config

type ApplicationRunner = Callable[[], int]


def load_application_runner() -> ApplicationRunner:
    """Resolve the UI entry point without importing application dependencies eagerly."""
    application = importlib.import_module("modules.app")
    runner = application.main
    if not callable(runner):
        raise TypeError("modules.app.main must be callable")
    return cast(ApplicationRunner, runner)


def collect_runtime_diagnostics(
    settings: Settings,
    *,
    check_ollama: bool = True,
) -> list[str]:
    """Return unmet runtime prerequisites without constructing service clients."""
    diagnostics: list[str] = []
    if check_ollama:
        try:
            with urlopen(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2):
                pass
        except (OSError, URLError):
            diagnostics.append(
                f"Ollama is unavailable at {settings.ollama_base_url}. "
                "Start it with `ollama serve` and pull the configured models."
            )
    return diagnostics


def _print_diagnostics(diagnostics: Sequence[str]) -> None:
    print("RAG application started with limited readiness:", file=sys.stderr)
    for diagnostic in diagnostics:
        print(f"- {diagnostic}", file=sys.stderr)


def main(
    *,
    settings: Settings = config,
    check_ollama: bool = True,
    app_runner: ApplicationRunner | None = None,
) -> int:
    """Report unavailable services, then lazily import and launch the application."""
    diagnostics = collect_runtime_diagnostics(settings, check_ollama=check_ollama)
    if diagnostics:
        _print_diagnostics(diagnostics)

    if app_runner is None:
        try:
            app_runner = load_application_runner()
        except ImportError as error:
            print(
                "RAG application dependencies are unavailable. Run `uv sync` and retry. "
                f"Import error: {error}",
                file=sys.stderr,
            )
            return 1

    return app_runner()


if __name__ == "__main__":
    raise SystemExit(main())
