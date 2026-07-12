"""Production middleware: request context, size limits, rate limiting, and API-key auth."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Header, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from lstm_forecast import __version__
from lstm_forecast.config import get_settings
from lstm_forecast.observability import get_logger, request_id_var

logger = get_logger("lstm_forecast.api")

_ONE_MINUTE = 60.0


def _error(status: int, detail: str, request_id: str) -> JSONResponse:
    """Consistent error envelope used across the API."""
    return JSONResponse(
        status_code=status,
        content={"error": True, "status": status, "detail": detail, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request, enforce a body-size cap, and log access.

    Adds ``X-Request-ID``, ``X-Process-Time-Ms`` and ``X-App-Version`` response headers, and
    binds the request id into the logging context so all logs for a request correlate.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        settings = get_settings()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            # Reject oversized bodies early (Content-Length based).
            cl = request.headers.get("content-length")
            if cl is not None and cl.isdigit() and int(cl) > settings.api.max_request_bytes:
                return _error(413, "Request body too large.", request_id)

            try:
                response = await call_next(request)
            except HTTPException:
                raise
            except Exception:
                logger.exception("Unhandled error handling %s %s", request.method, request.url.path)
                return _error(500, "Internal server error.", request_id)

            elapsed_ms = (time.perf_counter() - start) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
            response.headers["X-App-Version"] = __version__
            logger.info(
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            return response
        finally:
            request_id_var.reset(token)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter (per API key, else per client IP).

    Disabled when ``rate_limit_per_min`` is 0. In-process only — a multi-instance deployment
    would use a shared store (e.g. Redis) instead.
    """

    def __init__(self, app: object, limit_per_min: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.limit = limit_per_min
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if self.limit <= 0 or request.url.path in ("/health", "/ready"):
            return await call_next(request)
        client = request.headers.get("x-api-key") or (
            request.client.host if request.client else "anonymous"
        )
        now = time.monotonic()
        window = self._hits[client]
        while window and window[0] <= now - _ONE_MINUTE:
            window.popleft()
        if len(window) >= self.limit:
            request_id = request_id_var.get()
            resp = _error(429, "Rate limit exceeded. Try again shortly.", request_id)
            resp.headers["Retry-After"] = "60"
            return resp
        window.append(now)
        return await call_next(request)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce the configured API key when auth is enabled.

    A no-op when ``LSTM_FORECAST_API__API_KEY`` is unset (open/dev mode).
    """
    settings = get_settings()
    if not settings.api.auth_enabled:
        return
    if x_api_key != settings.api.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
