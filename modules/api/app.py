"""FastAPI application factory and lifecycle ownership."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from modules.api.dependencies import ApplicationContainer
from modules.api.errors import register_exception_handlers
from modules.api.routes import router

ContainerFactory = Callable[[], ApplicationContainer]


def create_app(container_factory: ContainerFactory) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = container_factory()
        await container.workspace.start()
        try:
            await container.benchmarks.start()
        except BaseException:
            await container.workspace.close()
            raise
        app.state.container = container
        try:
            yield
        finally:
            del app.state.container
            await container.benchmarks.close()
            await container.workspace.close()

    app = FastAPI(lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(router)
    return app
