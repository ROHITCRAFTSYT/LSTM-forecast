"""Tests for the async job queue and the trained-model cache (offline)."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from lstm_forecast.api import service
from lstm_forecast.api.main import create_app
from lstm_forecast.api.schemas import ForecastRequest


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _values_request(horizon=4, test_length=8, epochs=8, lags=8):
    rng = np.random.default_rng(0)
    series = (np.cumsum(rng.normal(0, 1, size=120)) + 100).tolist()
    return {
        "series": {"values": series},
        "horizon": horizon,
        "test_length": test_length,
        "lags": lags,
        "epochs": epochs,
    }


def test_submit_and_poll_job(client):
    submit = client.post("/jobs/forecast", json=_values_request())
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]
    assert isinstance(job_id, str) and job_id

    status = None
    body = None
    for _ in range(60):
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        status = body["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.25)

    assert status == "done", f"job did not finish cleanly: {body}"
    assert body["error"] is None
    assert body["result"] is not None
    assert len(body["result"]["forecast"]) == 4


def test_job_status_unknown_id(client):
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def test_run_forecast_cached_reuses_model():
    service.clear_model_cache()
    req = ForecastRequest(**_values_request())

    start = time.perf_counter()
    f1, result1 = service.run_forecast_cached(req)
    first_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    f2, result2 = service.run_forecast_cached(req)
    second_elapsed = time.perf_counter() - start

    # Same fitted Forecaster instance is reused on the second call.
    assert f2 is f1
    assert result2 is result1
    assert result2.point.size == req.horizon
    # The cached call avoids retraining, so it should not be slower than training.
    assert second_elapsed <= first_elapsed + 1.0

    service.clear_model_cache()


def test_model_cache_is_lru_bounded(monkeypatch):
    """The model cache evicts least-recently-used entries beyond its max size."""
    service.clear_model_cache()
    monkeypatch.setattr(service, "_MODEL_CACHE_MAXSIZE", 2)

    # Stub training so the test is fast and isolated from the model.
    monkeypatch.setattr(service, "run_forecast", lambda req: (object(), object()))

    def make_req(seed: int) -> ForecastRequest:
        rng = np.random.default_rng(seed)
        values = (np.cumsum(rng.normal(0, 1, size=40)) + 100).tolist()
        return ForecastRequest(
            series={"values": values}, horizon=3, test_length=6, lags=5, epochs=3
        )

    r1, r2, r3 = make_req(1), make_req(2), make_req(3)
    service.run_forecast_cached(r1)
    service.run_forecast_cached(r2)
    service.run_forecast_cached(r3)  # should evict r1 (oldest)

    assert len(service._MODEL_CACHE) == 2
    assert service.request_cache_key(r1) not in service._MODEL_CACHE
    assert service.request_cache_key(r2) in service._MODEL_CACHE
    assert service.request_cache_key(r3) in service._MODEL_CACHE
    service.clear_model_cache()


def test_model_cache_lru_touch_on_hit(monkeypatch):
    """Accessing an entry marks it most-recently-used so it survives eviction."""
    service.clear_model_cache()
    monkeypatch.setattr(service, "_MODEL_CACHE_MAXSIZE", 2)

    # On a cache hit, run_forecast_cached calls forecaster.forecast_future; give a stub.
    class _StubForecaster:
        def forecast_future(self, *, alpha):
            return None

    monkeypatch.setattr(service, "run_forecast", lambda req: (_StubForecaster(), object()))

    def make_req(seed: int) -> ForecastRequest:
        rng = np.random.default_rng(seed)
        values = (np.cumsum(rng.normal(0, 1, size=40)) + 100).tolist()
        return ForecastRequest(
            series={"values": values}, horizon=3, test_length=6, lags=5, epochs=3
        )

    r1, r2, r3 = make_req(1), make_req(2), make_req(3)
    service.run_forecast_cached(r1)
    service.run_forecast_cached(r2)
    service.run_forecast_cached(r1)  # touch r1 → now MRU
    service.run_forecast_cached(r3)  # should evict r2, not r1

    assert service.request_cache_key(r1) in service._MODEL_CACHE
    assert service.request_cache_key(r2) not in service._MODEL_CACHE
    service.clear_model_cache()
