"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from lstm_forecast.config import get_settings
from lstm_forecast.data import load_synthetic_prices


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Ensure each test sees a fresh settings instance."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def prices():
    """A small, deterministic synthetic OHLCV frame."""
    return load_synthetic_prices(n=300, seed=7)


@pytest.fixture
def trending_series():
    """A clean, learnable trend+seasonal series (signal the model should capture)."""
    t = np.arange(260)
    y = (
        100
        + 0.2 * t
        + 5 * np.sin(2 * np.pi * t / 20)
        + np.random.default_rng(0).normal(0, 0.5, t.size)
    )
    return y.astype(float)
