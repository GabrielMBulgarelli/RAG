from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from modules.api.app import create_app
from modules.api.dependencies import ApplicationContainer, BenchmarkManager, WorkspaceService
from modules.application.models import (
    CapabilitySnapshot,
    CorpusSnapshot,
    RuntimeSnapshot,
)


class StaticTestOwner:
    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def get_runtime(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            state="not_loaded",
            configured_chat_model="chat",
            active_chat_model=None,
            embedding_model="embed",
            available_chat_models=[],
            detail="Ready to load.",
            capabilities=CapabilitySnapshot(
                can_query=False,
                can_load_models=True,
                can_upload=False,
                can_run_benchmark=False,
            ),
            active_operation=None,
            corpus=CorpusSnapshot(
                document_count=0,
                page_count=0,
                chunk_count=0,
                status="empty",
            ),
        )


def make_container(owner: StaticTestOwner) -> ApplicationContainer:
    return ApplicationContainer(
        workspace=cast(WorkspaceService, owner),
        benchmarks=cast(BenchmarkManager, owner),
    )


def write_frontend_build(root: Path) -> Path:
    dist = root / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><script src="/assets/app.js"></script><main>RAG workspace</main>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("window.workspaceLoaded = true;", encoding="utf-8")
    return dist


def test_static_frontend_serves_root_and_assets_after_api_routes(tmp_path: Path) -> None:
    # Given a valid built frontend and lifecycle owner
    owner = StaticTestOwner()
    app = create_app(lambda: make_container(owner), frontend_dist=write_frontend_build(tmp_path))

    # When the browser requests both static and API resources
    with TestClient(app) as client:
        root = client.get("/")
        asset = client.get("/assets/app.js")
        runtime = client.get("/api/runtime")

    # Then each resource is served by its intended handler
    assert root.status_code == 200
    assert "RAG workspace" in root.text
    assert asset.status_code == 200
    assert asset.text == "window.workspaceLoaded = true;"
    assert runtime.status_code == 200
    assert runtime.json()["configured_chat_model"] == "chat"


def test_static_frontend_requires_an_index_document(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()

    with pytest.raises(FileNotFoundError, match="frontend production build"):
        create_app(lambda: make_container(StaticTestOwner()), frontend_dist=dist)
