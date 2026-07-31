"""Production FastAPI entry point for the React workspace."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from modules.api.app import create_app
from modules.bootstrap import create_application_container
from modules.config import PROJECT_ROOT, Settings, config

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
ServerRunner = Callable[..., Any]


def create_application(
    *,
    settings: Settings = config,
    frontend_dist: Path = FRONTEND_DIST,
) -> FastAPI:
    """Compose the production application while keeping services lifespan-owned."""
    return create_app(
        lambda: create_application_container(settings),
        frontend_dist=frontend_dist,
    )


def main(
    *,
    settings: Settings = config,
    frontend_dist: Path = FRONTEND_DIST,
    server_runner: ServerRunner = uvicorn.run,
) -> int:
    try:
        application = create_application(settings=settings, frontend_dist=frontend_dist)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    server_runner(
        application,
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
