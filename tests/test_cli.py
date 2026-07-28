"""CLI argument parsing.

The CLI is a primary entrypoint but was untested. These cover the parser wiring
(defaults, the new --json/--seed flags, and the serve subcommand) without running
a real forecast, which needs data and training.
"""

from __future__ import annotations

import pytest

from lstm_forecast.cli import build_parser


def test_forecast_defaults():
    ns = build_parser().parse_args(["forecast", "AAPL"])
    assert ns.command == "forecast"
    assert ns.ticker == "AAPL"
    assert ns.json is False
    assert ns.seed is None
    assert ns.horizon == 21


def test_forecast_json_and_seed_flags():
    ns = build_parser().parse_args(["forecast", "MSFT", "--json", "--seed", "7"])
    assert ns.json is True
    assert ns.seed == 7


def test_forecast_numeric_options_are_typed():
    ns = build_parser().parse_args(
        ["forecast", "X", "--horizon", "5", "--epochs", "3", "--alpha", "0.2"]
    )
    assert ns.horizon == 5 and ns.epochs == 3
    assert ns.alpha == pytest.approx(0.2)


def test_serve_subcommand():
    ns = build_parser().parse_args(["serve", "--port", "9000", "--reload"])
    assert ns.command == "serve"
    assert ns.port == 9000
    assert ns.reload is True


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
