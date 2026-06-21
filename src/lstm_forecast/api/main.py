"""FastAPI application factory: wiring, CORS, health, and routers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lstm_forecast import __version__
from lstm_forecast.ai.client import AIClient
from lstm_forecast.api.routes import chat, forecast, transfer
from lstm_forecast.api.schemas import HealthResponse
from lstm_forecast.config import get_settings
from lstm_forecast.utils import resolve_device


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="lstm-forecast",
        version=__version__,
        description=(
            "LSTM time-series forecasting for finance with retrieval-augmented forecasting "
            "and Claude-powered insights. Forecasts are uncertain and not financial advice."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    app.include_router(forecast.router)
    app.include_router(chat.router)
    app.include_router(transfer.router)
    return app


app = create_app()
