"""AI layer tests — exercising the no-key fallbacks and a mocked Claude path."""

from __future__ import annotations

import numpy as np

from lstm_forecast.ai import generate_insights, suggest_tuning
from lstm_forecast.ai.assistant import ChatAssistant
from lstm_forecast.ai.client import AIClient
from lstm_forecast.ai.doc_index import DocIndex
from lstm_forecast.config import AISettings
from lstm_forecast.forecasting.forecaster import Forecaster, ModelSpec


def _result(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=5, test_length=10)
    return f.fit_predict(ModelSpec(lags=10, hidden_size=16, epochs=10, patience=4))


def test_client_unavailable_without_key():
    client = AIClient(AISettings(api_key=""))
    assert client.available is False


def test_generate_insights_fallback_offline(prices):
    res = _result(prices)
    text = generate_insights(res, label="TEST", client=AIClient(AISettings(api_key="")))
    assert "offline mode" in text.lower()
    assert "forecast" in text.lower()


def test_suggest_tuning_fallback_offline():
    sug = suggest_tuning(np.linspace(100, 120, 200), client=AIClient(AISettings(api_key="")))
    assert len(sug.candidates) >= 1
    assert sug.candidates[0].lags >= 2


def test_doc_index_search():
    idx = DocIndex()
    idx.add_text("The LSTM model uses attention over hidden states.", source="docs")
    idx.add_text("Conformal prediction provides coverage guarantees.", source="docs")
    idx.build()
    hits = idx.search("attention", k=1)
    assert hits and "attention" in hits[0].text.lower()


def test_chat_assistant_fallback(prices):
    idx = DocIndex()
    idx.add_text("Conformal intervals are calibrated from residuals.", source="docs")
    idx.build()
    res = _result(prices)
    assistant = ChatAssistant(idx, result=res, client=AIClient(AISettings(api_key="")))
    answer = assistant.ask("How are intervals calibrated?")
    assert "offline" in answer.lower()
    assert "conformal" in answer.lower()


def test_generate_insights_with_mocked_llm(prices, monkeypatch):
    res = _result(prices)
    client = AIClient(AISettings(api_key="test-key"))  # anthropic SDK installed → available
    assert client.available
    # Stub the network call at the client boundary (provider-agnostic).
    monkeypatch.setattr(
        AIClient, "complete", lambda self, **kw: "LLM says: the series trends up modestly."
    )
    text = generate_insights(res, label="TEST", client=client)
    assert "LLM says" in text


def test_provider_selection_openai_and_ollama():
    from lstm_forecast.ai.providers import OpenAIProvider, build_provider

    openai_p = build_provider(AISettings(provider="openai", api_key="k"))
    assert isinstance(openai_p, OpenAIProvider)
    # Ollama is keyless and OpenAI-compatible; enabled even with no key.
    ollama = AISettings(provider="ollama", api_key="")
    assert ollama.enabled is True
    assert build_provider(ollama).name == "openai"


def test_resolved_model_defaults_per_provider():
    assert AISettings(provider="anthropic").resolved_model == "claude-opus-4-8"
    assert AISettings(provider="openai").resolved_model == "gpt-4o"
    assert AISettings(provider="google").resolved_model.startswith("gemini")
    assert AISettings(provider="openai", model="gpt-4o-mini").resolved_model == "gpt-4o-mini"
