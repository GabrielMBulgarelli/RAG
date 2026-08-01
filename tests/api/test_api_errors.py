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
        ("index_error", 503),
        ("benchmark_unavailable", 503),
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
    body = response.json()
    errors = body["details"]["errors"]
    assert body == {
        "code": "request_validation_failed",
        "message": "Request validation failed.",
        "details": {"errors": errors},
    }
    assert {(tuple(error["loc"]), error["type"], error["input"]) for error in errors} >= {
        (("body", "session_id"), "uuid_parsing", "not-a-uuid"),
        (("body", "question"), "value_error", " \t "),
    }
    json.dumps(body["details"])


def test_invalid_last_event_id_uses_normalized_validation_body() -> None:
    app = error_test_app()
    run_id = "4cbdbcb9-5a57-4514-a392-2dce907456d5"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            f"/api/benchmarks/{run_id}/events",
            headers={"Last-Event-ID": "not-an-integer"},
        )

    assert response.status_code == 422
    body = response.json()
    errors = body["details"]["errors"]
    assert body == {
        "code": "request_validation_failed",
        "message": "Request validation failed.",
        "details": {"errors": errors},
    }
    assert {(tuple(error["loc"]), error["type"], error["input"]) for error in errors} == {
        (("header", "Last-Event-ID"), "int_parsing", "not-an-integer"),
    }
    json.dumps(body["details"])


def test_negative_last_event_id_uses_normalized_validation_body() -> None:
    app = error_test_app()
    run_id = "4cbdbcb9-5a57-4514-a392-2dce907456d5"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            f"/api/benchmarks/{run_id}/events",
            headers={"Last-Event-ID": "-1"},
        )

    assert response.status_code == 422
    body = response.json()
    errors = body["details"]["errors"]
    assert body == {
        "code": "request_validation_failed",
        "message": "Request validation failed.",
        "details": {"errors": errors},
    }
    assert {(tuple(error["loc"]), error["type"], error["input"]) for error in errors} == {
        (("header", "Last-Event-ID"), "greater_than_equal", "-1"),
    }


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
