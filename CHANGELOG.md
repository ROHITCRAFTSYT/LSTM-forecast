# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-06-21

### Added
- **Provider-agnostic AI layer** — use any LLM, not just Claude: `anthropic`, `openai`,
  `google` (Gemini), `ollama` (local, no key), or `openai_compatible` (OpenRouter, Together,
  Groq, vLLM, …). Selected via `LSTM_FORECAST_AI__PROVIDER`; structured output and the chat
  assistant work uniformly across providers; Anthropic remains the default. New extras
  `ai-openai`, `ai-google`, `ai-all`.
- **Model persistence** — `Forecaster.save(path)` / `Forecaster.load(path)` +
  `forecast_future()` produce forecasts from a saved ensemble with **no retraining**.
- **Cross-validated tuning** — `Forecaster.tune(specs)` evaluates a candidate grid with
  walk-forward CV and adopts the winner; `forecasting.tuning.specs_from_suggestion` converts
  an AI `TuningSuggestion` into specs, closing the "LLM proposes, data decides" loop.
- **Ensemble forecasting** — `ModelSpec.ensemble=N` trains N differently-seeded models and
  averages them for lower-variance, better-calibrated point forecasts.
- **Diebold–Mariano significance test** (`evaluation.significance`) — every benchmarked run
  reports whether the model is *statistically* better than naive (HLN small-sample corrected);
  surfaced in `ForecastResult.significance`, the API response, the CLI, and AI insights.
- **CLI** — `forecast --tune` (CV the AI grid) and `forecast --ensemble N` flags.

### Changed
- The model's own test metric is now always computed (even with `benchmark=False`), enabling CV.

## [0.1.0] - 2026-06-21

### Added
- **Forecasting core** — custom PyTorch LSTM with optional additive attention and
  point/quantile heads; direct multi-horizon prediction in delta (change-from-last) space.
- **Reversible transforms** — leakage-safe `Detrend`, `Deseason`, `RobustScale`,
  `StandardScale`, `Log`, `Difference`, composed via `Transformer`/`Reverter`; correct
  inversion at future positions.
- **Public API** — `Forecaster` (test-set benchmarking, conformal intervals, transfer
  learning) and `Pipeline` composer mirroring the scalecast ergonomics.
- **Probabilistic forecasting** — split-conformal static intervals and backtest
  residual-matrix dynamic (horizon-aware) intervals.
- **Baselines** — Naive, Drift, Seasonal-Naive, ARIMA, ETS for honest benchmarking.
- **Retrieval-augmented forecasting (RAG)** — z-normalised analog-window index (FAISS with
  NumPy fallback) producing per-timestep analog feature channels.
- **Claude AI layer** — natural-language insights, structured LLM-assisted tuning, and a
  RAG chat assistant; all degrade gracefully with no API key.
- **Finance features** — log-returns, volatility, RSI, MACD, Bollinger, calendar & Fourier.
- **Surfaces** — FastAPI service (`/forecast`, `/backtest`, `/insights`, `/chat`,
  `/transfer`, `/health`), Streamlit dashboard, and a CLI (`lstm-forecast`).
- **Packaging & ops** — pip-installable package with optional extras, Dockerfiles +
  compose, GitHub Actions CI (lint/type/test matrix + image build), pre-commit, mkdocs docs,
  runnable examples for the article's five capabilities, and a model card.
