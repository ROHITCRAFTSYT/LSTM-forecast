"""FastAPI application factory: lifespan, middleware, error handling, health, and routers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from lstm_forecast import __version__
from lstm_forecast.ai.client import AIClient
from lstm_forecast.api.jobs import get_job_manager
from lstm_forecast.api.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    require_api_key,
)
from lstm_forecast.api.routes import chat, forecast, jobs, transfer
from lstm_forecast.api.schemas import HealthResponse, ReadyResponse
from lstm_forecast.config import get_settings
from lstm_forecast.observability import configure_logging, get_logger, request_id_var
from lstm_forecast.utils import resolve_device


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on startup and shut the job manager down cleanly on exit."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger = get_logger("lstm_forecast.api")
    logger.info(
        "lstm-forecast API v%s starting (device=%s, ai_provider=%s)",
        __version__,
        resolve_device(settings.device),
        settings.ai.provider,
    )
    try:
        yield
    finally:
        get_job_manager().shutdown()
        logger.info("lstm-forecast API shut down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="lstm-forecast",
        version=__version__,
        description=(
            "Provider-agnostic LSTM time-series forecasting for finance with "
            "retrieval-augmented forecasting, conformal uncertainty, and an LLM insight "
            "layer. Forecasts are uncertain and not financial advice."
        ),
        lifespan=lifespan,
    )

    # Middleware — added last runs outermost, so request-context wraps everything.
    if settings.api.rate_limit_per_min > 0:
        app.add_middleware(RateLimitMiddleware, limit_per_min=settings.api.rate_limit_per_min)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    _register_error_handlers(app)

    # Public probes (no auth, no rate limit).
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        client = AIClient()
        return HealthResponse(
            status="ok",
            version=__version__,
            ai_enabled=client.available,
            ai_provider=settings.ai.provider,
            device=resolve_device(settings.device),
        )

    @app.get("/ready", response_model=ReadyResponse, tags=["meta"])
    def ready() -> ReadyResponse:
        # Readiness = the heavy dependency (torch) imports and a device resolves.
        checks: dict[str, bool] = {}
        try:
            import torch  # noqa: F401

            checks["torch"] = True
        except Exception:
            checks["torch"] = False
        checks["auth_required"] = settings.api.auth_enabled
        return ReadyResponse(ready=all(v for k, v in checks.items() if k == "torch"), checks=checks)

    # Feature routers require the API key when auth is enabled.
    auth = [Depends(require_api_key)]
    app.include_router(forecast.router, dependencies=auth)
    app.include_router(chat.router, dependencies=auth)
    app.include_router(transfer.router, dependencies=auth)
    app.include_router(jobs.router, dependencies=auth)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Return a consistent JSON error envelope for every failure mode."""

    def envelope(status: int, detail: str) -> JSONResponse:
        rid = request_id_var.get()
        return JSONResponse(
            status_code=status,
            content={"error": True, "status": status, "detail": detail, "request_id": rid},
            headers={"X-Request-ID": rid},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return envelope(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return envelope(
            422, "; ".join(e.get("msg", "invalid") for e in exc.errors()) or "Invalid request."
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        get_logger("lstm_forecast.api").exception("Unhandled exception: %s", exc)
        return envelope(500, "Internal server error.")


app = create_app()
