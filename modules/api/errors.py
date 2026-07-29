"""Centralized HTTP error normalization."""

import json

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from modules.application.errors import ApplicationError
from modules.application.models import ApiProblem

APPLICATION_ERROR_STATUS = {
    "operation_busy": 409,
    "document_not_found": 404,
    "conversation_not_found": 404,
    "benchmark_not_found": 404,
    "upload_limit_exceeded": 413,
    "invalid_upload": 400,
    "invalid_request": 400,
    "runtime_unavailable": 503,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        _request: Request,
        exception: ApplicationError,
    ) -> JSONResponse:
        problem = ApiProblem(
            code=exception.code,
            message=exception.message,
            details=exception.details,
        )
        return JSONResponse(
            status_code=APPLICATION_ERROR_STATUS.get(exception.code, 400),
            content=problem.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        _request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        errors = json.loads(json.dumps(exception.errors(), default=str))
        problem = ApiProblem(
            code="request_validation_failed",
            message="Request validation failed.",
            details={"errors": errors},
        )
        return JSONResponse(
            status_code=422,
            content=problem.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request,
        _exception: Exception,
    ) -> JSONResponse:
        problem = ApiProblem(
            code="internal_error",
            message="An unexpected internal error occurred.",
            details={},
        )
        return JSONResponse(
            status_code=500,
            content=problem.model_dump(mode="json"),
        )
