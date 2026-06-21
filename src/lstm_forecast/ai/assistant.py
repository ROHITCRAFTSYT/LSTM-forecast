"""A provider-agnostic RAG chat assistant grounded in docs and the current forecast run.

Uses a portable retrieve-then-answer pattern (retrieve relevant context, then ask the model
to answer grounded in it) so it works identically across Anthropic, OpenAI, Gemini, Ollama,
and OpenAI-compatible endpoints. Without a usable provider it returns the retrieved docs.
"""

from __future__ import annotations

import json

from lstm_forecast.ai.client import AIClient
from lstm_forecast.ai.doc_index import DocIndex
from lstm_forecast.forecasting.forecaster import ForecastResult

_SYSTEM = (
    "You are an assistant for the lstm-forecast library. Answer the user's question using "
    "ONLY the provided context (documentation excerpts and the current forecast run). If the "
    "context doesn't cover it, say so. Be concise and never give financial advice."
)


class ChatAssistant:
    """Grounded chat over docs + the active :class:`ForecastResult`, any LLM provider."""

    def __init__(
        self,
        doc_index: DocIndex,
        *,
        result: ForecastResult | None = None,
        client: AIClient | None = None,
    ) -> None:
        self.doc_index = doc_index
        self.result = result
        self.client = client or AIClient()

    def _run_context(self) -> str:
        if self.result is None:
            return ""
        parts = ["Current forecast run:"]
        if self.result.metrics:
            parts.append("Test-set metrics (RMSE-sorted):\n" + self.result.metrics_frame().to_string())
        if self.result.significance.get("vs_naive"):
            parts.append("Significance vs naive: " + json.dumps(self.result.significance["vs_naive"]))
        parts.append(
            "Forecast point path: "
            + json.dumps([round(v, 4) for v in self.result.point.tolist()])
        )
        return "\n".join(parts)

    def _context(self, question: str) -> str:
        chunks = self.doc_index.search(question, k=4)
        doc_text = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks) or "(no docs matched)"
        blocks = [f"Documentation:\n{doc_text}"]
        run = self._run_context()
        if run:
            blocks.append(run)
        return "\n\n".join(blocks)

    def ask(self, question: str) -> str:
        """Answer ``question`` grounded in retrieved context (LLM if available, else docs)."""
        context = self._context(question)
        if not self.client.available:
            return self._fallback(question, context)
        user = f"Context:\n{context}\n\nQuestion: {question}"
        try:
            return self.client.complete(system=_SYSTEM, messages=[{"role": "user", "content": user}])
        except Exception:
            return self._fallback(question, context)

    def _fallback(self, question: str, context: str) -> str:
        chunks = self.doc_index.search(question, k=3)
        if not chunks:
            return (
                "AI chat is offline (no LLM provider configured) and no relevant documentation "
                "was found. Configure a provider to enable grounded answers."
            )
        body = "\n\n".join(f"- {c.text}" for c in chunks)
        return (
            "AI chat is offline (no LLM provider configured); returning the most relevant "
            f"documentation for your question:\n\n{body}"
        )
