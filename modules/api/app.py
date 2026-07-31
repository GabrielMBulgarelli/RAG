"""FastAPI application factory and lifecycle ownership."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from modules.api.dependencies import ApplicationContainer
from modules.api.errors import register_exception_handlers
from modules.api.routes import router

ContainerFactory = Callable[[], ApplicationContainer]


def create_app(
    container_factory: ContainerFactory,
    *,
    frontend_dist: Path | None = None,
) -> FastAPI:
    if frontend_dist is not None and not (frontend_dist / "index.html").is_file():
        raise FileNotFoundError(
            f"Expected a frontend production build at {frontend_dist}. "
            "Run `cd frontend && npm ci && npm run build` first."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = container_factory()
        try:
            await container.workspace.start()
        except BaseException:
            await container.workspace.close()
            raise
        try:
            await container.benchmarks.start()
        except BaseException:
            try:
                await container.benchmarks.close()
            finally:
                await container.workspace.close()
            raise
        app.state.container = container
        try:
            yield
        finally:
            del app.state.container
            try:
                await container.benchmarks.close()
            finally:
                await container.workspace.close()

    app = FastAPI(lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(router)
    if frontend_dist is not None:
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app
