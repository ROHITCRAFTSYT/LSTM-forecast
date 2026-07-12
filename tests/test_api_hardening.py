"""Production-hardening tests: headers, error envelope, readiness, auth, rate limit, size."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from lstm_forecast.api.main import create_app
from lstm_forecast.api.middleware import RateLimitMiddleware, require_api_key
from lstm_forecast.config import get_settings


@pytest.fixture
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def test_request_context_headers(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("x-request-id")
    assert r.headers.get("x-app-version")
    assert r.headers.get("x-process-time-ms")


def test_ready_endpoint(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["checks"]["torch"] is True


def test_error_envelope_shape(client):
    r = client.post("/forecast", json={"series": {}})  # invalid
    assert r.status_code == 422
    body = r.json()
    assert body["error"] is True
    assert set(body) >= {"error", "status", "detail", "request_id"}
    assert r.headers.get("x-request-id")


def test_api_key_auth_blocks_without_key(monkeypatch):
    monkeypatch.setenv("LSTM_FORECAST_API__API_KEY", "s3cret")
    get_settings.cache_clear()
    c = TestClient(create_app(), raise_server_exceptions=False)
    # Public probe stays open.
    assert c.get("/health").status_code == 200
    # Feature route is blocked before any training happens.
    r = c.post("/forecast", json={"series": {"values": [1.0, 2.0, 3.0]}})
    assert r.status_code == 401


def test_require_api_key_dependency(monkeypatch):
    monkeypatch.setenv("LSTM_FORECAST_API__API_KEY", "s3cret")
    get_settings.cache_clear()
    # Correct key passes (returns None), wrong key raises 401.
    assert asyncio.run(require_api_key("s3cret")) is None
    with pytest.raises(HTTPException) as exc:
        asyncio.run(require_api_key("wrong"))
    assert exc.value.status_code == 401


def test_rate_limit_middleware():
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limit_per_min=2)

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    c = TestClient(app)
    assert c.get("/ping").status_code == 200
    assert c.get("/ping").status_code == 200
    r = c.get("/ping")
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "60"


def test_max_request_body(monkeypatch):
    monkeypatch.setenv("LSTM_FORECAST_API__MAX_REQUEST_BYTES", "1024")
    get_settings.cache_clear()
    c = TestClient(create_app(), raise_server_exceptions=False)
    big = {"series": {"values": [float(i) for i in range(500)]}}  # body well over 1 KB
    r = c.post("/forecast", json=big)
    assert r.status_code == 413
