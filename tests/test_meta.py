"""Packaging/metadata tests."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import lstm_forecast


def test_version_is_single_sourced():
    # __version__ comes from installed package metadata, not a hardcoded string.
    assert lstm_forecast.__version__ == version("lstm-forecast")
    assert lstm_forecast.__version__ != "0.0.0+unknown"


def test_py_typed_marker_present():
    marker = Path(lstm_forecast.__file__).parent / "py.typed"
    assert marker.exists(), "PEP 561 py.typed marker must ship with the package"
