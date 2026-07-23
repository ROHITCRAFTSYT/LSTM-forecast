"""Forecast, backtest and insights endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lstm_forecast.ai.insights import generate_insights
from lstm_forecast.api import service
from lstm_forecast.api.schemas import (
    BacktestRequest,
    ForecastRequest,
    ForecastResponse,
)

router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    """Train, evaluate against baselines, and forecast the future with conformal intervals."""
    try:
        _, result = service.run_forecast(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    insights = (
        generate_insights(result, label=req.series.ticker or "series")
        if req.include_insights
        else None
    )
    return service.to_response(req, result, insights=insights)


@router.post("/backtest", response_model=ForecastResponse)
def backtest(req: BacktestRequest) -> ForecastResponse:
    """Forecast with horizon-aware dynamic intervals from a rolling-origin backtest."""
    try:
        _, result = service.run_forecast(
            req, run_backtest=True, backtest_windows=req.backtest_windows
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    insights = (
        generate_insights(result, label=req.series.ticker or "series")
        if req.include_insights
        else None
    )
    return service.to_response(req, result, insights=insights)


@router.post("/insights")
def insights(req: ForecastRequest) -> dict[str, str]:
    """Run a forecast and return only the AI narrative (Claude if configured, else template)."""
    try:
        _, result = service.run_forecast(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"insights": generate_insights(result, label=req.series.ticker or "series")}
