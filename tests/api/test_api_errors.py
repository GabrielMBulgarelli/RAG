import json
from typing import cast

import pytest
from fastapi.testclient import TestClient

from modules.api.app import create_app
from modules.api.dependencies import (
    ApplicationContainer,
    BenchmarkManager,
    WorkspaceService,
)
from modules.application.errors import ApplicationError


class NoopOwner:
    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


def error_test_app():
    owner = NoopOwner()
    return create_app(
        lambda: ApplicationContainer(
            workspace=cast(WorkspaceService, owner),
            benchmarks=cast(BenchmarkManager, owner),
        )
    )


@pytest.mark.parametrize(
    ("code", "status_code"),
    [
        ("operation_busy", 409),
        ("document_not_found", 404),
        ("conversation_not_found", 404),
        ("benchmark_not_found", 404),
        ("upload_limit_exceeded", 413),
        ("invalid_upload", 400),
        ("invalid_request", 400),
        ("runtime_unavailable", 503),
        ("unknown_application_error", 400),
    ],
)
def test_application_errors_are_normalized_with_known_status_mapping(
    code: str,
    status_code: int,
) -> None:
    app = error_test_app()

    async def raise_application_error() -> None:
        raise ApplicationError(
            code=code,
            message="Safe message.",
            details={"field": "question"},
        )

    app.add_api_route("/test-error", raise_application_error, methods=["GET"])

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test-error")

    assert response.status_code == status_code
    assert response.json() == {
        "code": code,
        "message": "Safe message.",
        "details": {"field": "question"},
    }


def test_request_validation_errors_have_json_safe_structured_details() -> None:
    app = error_test_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/query",
            json={"session_id": "not-a-uuid", "question": " \t "},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_failed"
    assert response.json()["message"] == "Request validation failed."
    assert isinstance(response.json()["details"]["errors"], list)
    json.dumps(response.json()["details"])


def test_unexpected_errors_are_sanitized_without_exception_text() -> None:
    app = error_test_app()

    async def raise_unexpected_error() -> None:
        raise RuntimeError("secret database path")

    app.add_api_route("/unexpected-error", raise_unexpected_error, methods=["GET"])

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/unexpected-error")

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "An unexpected internal error occurred.",
        "details": {},
    }
    assert "secret database path" not in response.text
