"""LLM-assisted tuning: Claude proposes a candidate config grid; the data decides.

Claude returns a *structured* set of candidate configurations (validated via Pydantic). The
caller is expected to cross-validate / backtest these candidates and pick the winner — the
LLM suggests, it does not get the final say. Without an API key, a sensible default grid is
returned so the workflow still functions.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from lstm_forecast.ai.client import AIClient


class CandidateConfig(BaseModel):
    """One candidate model configuration proposed by the tuner."""

    lags: int = Field(ge=2, le=512)
    hidden_size: int = Field(ge=4, le=512)
    num_layers: int = Field(ge=1, le=4)
    dropout: float = Field(ge=0.0, le=0.6)
    use_attention: bool = True
    rationale: str = ""


class TuningSuggestion(BaseModel):
    """A structured tuning suggestion: transforms + a candidate config grid."""

    recommended_transforms: list[str] = Field(default_factory=list)
    candidates: list[CandidateConfig] = Field(default_factory=list)
    notes: str = ""


_SYSTEM = (
    "You are an expert at configuring LSTM time-series forecasters for financial data. "
    "Given summary statistics of a series, propose a small grid (3-6) of sensible candidate "
    "hyperparameter configurations to cross-validate, and recommend reversible transforms "
    "from this set only: ['detrend', 'deseason', 'robust_scale', 'log', 'difference']. "
    "Favour robustness for noisy financial series. Return concise rationales."
)


def _series_summary(y: np.ndarray) -> str:
    y = np.asarray(y, dtype=float).ravel()
    rets = np.diff(np.log(np.clip(y, 1e-9, None)))
    return (
        f"n={y.size}, mean={y.mean():.4f}, std={y.std():.4f}, "
        f"min={y.min():.4f}, max={y.max():.4f}, "
        f"approx_trend_slope={np.polyfit(np.arange(y.size), y, 1)[0]:.6f}, "
        f"daily_log_return_vol={rets.std():.5f}"
    )


def _default_suggestion() -> TuningSuggestion:
    return TuningSuggestion(
        recommended_transforms=["detrend", "robust_scale"],
        candidates=[
            CandidateConfig(
                lags=21, hidden_size=64, num_layers=1, dropout=0.1, rationale="balanced default"
            ),
            CandidateConfig(
                lags=42,
                hidden_size=64,
                num_layers=2,
                dropout=0.15,
                rationale="longer memory, more depth",
            ),
            CandidateConfig(
                lags=10,
                hidden_size=32,
                num_layers=1,
                dropout=0.05,
                rationale="light model for short/noisy series",
            ),
        ],
        notes="Offline default grid (no ANTHROPIC_API_KEY). Cross-validate these candidates.",
    )


def suggest_tuning(y: np.ndarray, *, client: AIClient | None = None) -> TuningSuggestion:
    """Return a structured tuning suggestion (Claude if available, else a default grid)."""
    client = client or AIClient()
    if not client.available:
        return _default_suggestion()
    user = (
        "Series summary statistics:\n"
        f"{_series_summary(y)}\n\n"
        "Propose transforms and a candidate hyperparameter grid to cross-validate."
    )
    try:
        return client.parse(system=_SYSTEM, user=user, schema=TuningSuggestion)
    except Exception:
        return _default_suggestion()
