"""Command-line interface: run a forecast or launch the API from the terminal."""

from __future__ import annotations

import argparse
import sys

from lstm_forecast.config import get_settings


def _cmd_forecast(args: argparse.Namespace) -> int:
    from lstm_forecast import Forecaster
    from lstm_forecast.ai import generate_insights
    from lstm_forecast.data import add_finance_features, load_prices
    from lstm_forecast.transforms import default_finance_transformer

    if args.seed is not None:
        from lstm_forecast.utils import set_seed

        set_seed(args.seed)

    df = load_prices(args.ticker, allow_synthetic_fallback=args.allow_synthetic)
    feat = add_finance_features(df, fourier_periods=(5.0,)) if args.features else df
    exog = feat.drop(columns=["close"]) if args.features else None

    f = Forecaster(
        y=feat["close"],
        current_dates=feat.index,
        future_dates=args.horizon,
        test_length=args.test_length,
        exog=exog,
        name="lstm",
    )
    transformer, reverter = default_finance_transformer(seasonal_period=args.seasonal_period)
    f.attach_transformer(transformer, reverter)

    from lstm_forecast.forecasting.forecaster import ModelSpec

    spec = ModelSpec(lags=args.lags, epochs=args.epochs, ensemble=args.ensemble)
    if args.tune:
        from lstm_forecast.ai import suggest_tuning
        from lstm_forecast.forecasting.tuning import specs_from_suggestion

        print("Cross-validating AI-suggested hyperparameters…")
        specs = specs_from_suggestion(suggest_tuning(f.y), base=spec)
        report = f.tune(specs, k=2)
        spec = f.spec  # tune() adopts the best candidate on the forecaster
        print(
            f"  best: lags={spec.lags} hidden={spec.hidden_size} "
            f"layers={spec.num_layers} (CV RMSE={report['best_cv_rmse']})"
        )

    result = f.fit_predict(spec, alpha=args.alpha)

    if args.json:
        import json

        payload = {"ticker": args.ticker, **result.to_dict()}
        if args.insights:
            payload["insights"] = generate_insights(result, label=args.ticker)
        print(json.dumps(payload, default=str))
        if args.plot:
            import matplotlib.pyplot as plt

            f.plot(ci=True)
            out = args.plot if isinstance(args.plot, str) else f"{args.ticker}_forecast.png"
            plt.tight_layout()
            plt.savefig(out, dpi=120)
        return 0

    print(f"\n=== {args.ticker} — test-set benchmark (RMSE-sorted) ===")
    print(result.metrics_frame().to_string())
    dm = result.significance.get("vs_naive")
    if isinstance(dm, dict):
        sig = "significant" if dm.get("significant") else "not significant"
        print(
            f"\nDiebold-Mariano vs naive: winner={dm.get('winner')}, "
            f"p={dm.get('p_value'):.3f} ({sig} at 5%)"
        )
    print(f"\n=== {args.horizon}-step forecast ===")
    for d, p, lo, hi in zip(
        result.future_dates, result.point, result.lower, result.upper, strict=False
    ):
        print(f"  {str(d)[:10]}  {p:10.4f}  [{lo:10.4f}, {hi:10.4f}]")

    if args.insights:
        print("\n=== AI insights ===")
        print(generate_insights(result, label=args.ticker))

    if args.plot:
        import matplotlib.pyplot as plt

        f.plot(ci=True)
        out = args.plot if isinstance(args.plot, str) else f"{args.ticker}_forecast.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"\nSaved plot to {out}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "lstm_forecast.api.main:app",
        host=args.host or settings.api.host,
        port=args.port or settings.api.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lstm-forecast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fc = sub.add_parser("forecast", help="Run a forecast for a ticker.")
    fc.add_argument("ticker")
    fc.add_argument("--horizon", type=int, default=21)
    fc.add_argument("--test-length", type=int, default=42)
    fc.add_argument("--lags", type=int, default=21)
    fc.add_argument("--epochs", type=int, default=60)
    fc.add_argument("--ensemble", type=int, default=1, help="Number of seeded models to average.")
    fc.add_argument(
        "--tune",
        action="store_true",
        help="Cross-validate AI-suggested hyperparameters before forecasting.",
    )
    fc.add_argument("--alpha", type=float, default=0.1)
    fc.add_argument("--seasonal-period", type=int, default=5)
    fc.add_argument("--features", action="store_true", help="Add finance features (multivariate).")
    fc.add_argument("--insights", action="store_true", help="Generate AI insights.")
    fc.add_argument(
        "--json",
        action="store_true",
        help="Emit the result as a single JSON object (for piping/automation).",
    )
    fc.add_argument(
        "--seed", type=int, default=None, help="Seed all RNGs for a reproducible run."
    )
    fc.add_argument("--plot", nargs="?", const=True, default=False, help="Save a forecast plot.")
    fc.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Fall back to synthetic data if the provider is unavailable.",
    )
    fc.set_defaults(func=_cmd_forecast)

    sv = sub.add_parser("serve", help="Launch the FastAPI service.")
    sv.add_argument("--host", default=None)
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
