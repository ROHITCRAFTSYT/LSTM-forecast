"""Structured logging and a request-id context for the whole application.

Framework-agnostic (standard library ``logging`` only). ``configure_logging`` sets up a
root handler with either a human-readable text format or line-delimited JSON, and every log
record automatically carries the current request id (set per-request by the API middleware),
so logs can be correlated across a request without threading an id through call sites.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar

# Per-request correlation id; "-" when outside a request (e.g. CLI, startup).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_CONFIGURED = False


class _RequestIdFilter(logging.Filter):
    """Attach the current request id to every record as ``record.request_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    """Minimal line-delimited JSON formatter (dependency-free)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure root logging once. Safe to call multiple times (idempotent)."""
    global _CONFIGURED
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.addFilter(_RequestIdFilter())
    if fmt.lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root = logging.getLogger()
    # Replace prior handlers so repeated calls (tests, reload) don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (configures logging with defaults if not yet configured)."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
