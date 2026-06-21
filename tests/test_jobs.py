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
