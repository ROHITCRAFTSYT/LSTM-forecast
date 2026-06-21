"""The public :class:`Forecaster` — the headline API users will copy.

It mirrors the ergonomics of the scalecast ``Forecaster`` referenced in the design doc
(``test_length``, ``future_dates``, ``plot(ci=True)``, ``transfer_predict``) but the engine
is the custom PyTorch model with attention and probabilistic heads, plus optional
retrieval-augmented features, conformal intervals, baseline benchmarking and backtesting.

Flow on ``fit_predict``:

1.  **Test evaluation** — fit the transformer on the training portion only, train a model
    with horizon = ``test_length``, forecast the test window, revert to the original scale,
    and benchmark against baselines. This is the honest "did we beat naive/ARIMA?" check.
2.  **Production forecast** — refit the transformer on *all* data, train a model with
    horizon = ``future_dates``, and forecast the future.
3.  **Intervals** — size static conformal intervals from the test residuals; optionally
    compute horizon-aware dynamic intervals via backtesting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lstm_forecast.evaluation.metrics import interval_metrics, point_metrics
from lstm_forecast.evaluation.significance import diebold_mariano
from lstm_forecast.forecasting.backtest import BacktestResult, backtest, dynamic_intervals
from lstm_forecast.forecasting.baselines import baseline_registry
from lstm_forecast.forecasting.conformal import conformal_intervals, conformal_quantile
from lstm_forecast.models.dataset import last_input_window, make_windows
from lstm_forecast.models.lstm import LSTMForecaster
from lstm_forecast.models.trainer import Trainer, TrainerConfig
from lstm_forecast.utils import resolve_device, set_seed

if TYPE_CHECKING:
    from lstm_forecast.rag.retriever import AnalogRetriever
    from lstm_forecast.transforms.pipeline import Reverter, Transformer


@dataclass
class ModelSpec:
    """Model + training hyperparameters used for every internal training pass."""

    lags: int = 21
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    use_attention: bool = True
    quantiles: list[float] | None = None
    target_mode: str = "delta"  # "delta" (predict change from last value) or "level"
    ensemble: int = 1  # number of differently-seeded models to average (>=1)
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 10
    val_fraction: float = 0.2


@dataclass
class ForecastResult:
    """Everything produced by a forecast run, in the original data scale."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    future_dates: pd.Index
    history_dates: pd.Index
    history_values: np.ndarray
    test_dates: pd.Index | None = None
    test_actual: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    interval: dict[str, float] = field(default_factory=dict)
    significance: dict[str, object] = field(default_factory=dict)
    backtest_result: BacktestResult | None = None
    alpha: float = 0.1

    def metrics_frame(self) -> pd.DataFrame:
        """Model-vs-baseline metrics as a tidy, RMSE-sorted DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        frame = pd.DataFrame(self.metrics).T
        return frame.sort_values("rmse") if "rmse" in frame else frame

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (for the API layer)."""
        return {
            "future_dates": [d.isoformat() for d in self.future_dates],
            "point": self.point.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "alpha": self.alpha,
            "metrics": self.metrics,
            "interval": self.interval,
            "test": self._test_dict(),
        }

    def _test_dict(self) -> dict[str, Any] | None:
        if self.test_actual is None or self.test_pred is None or self.test_dates is None:
            return None
        return {
            "dates": [d.isoformat() for d in self.test_dates],
            "actual": self.test_actual.tolist(),
            "pred": self.test_pred.tolist(),
        }


class Forecaster:
    """Fit, evaluate and forecast a single (optionally multivariate) time series."""

    def __init__(
        self,
        *,
        y: np.ndarray | pd.Series,
        current_dates: pd.Index | None = None,
        future_dates: int = 21,
        test_length: int = 42,
        exog: pd.DataFrame | None = None,
        name: str = "lstm",
        seed: int = 20,
        device: str = "auto",
    ) -> None:
        if isinstance(y, pd.Series):
            if current_dates is None:
                current_dates = y.index
            y = y.to_numpy()
        self.y = np.asarray(y, dtype=float).ravel()
        n = self.y.size
        self.dates: pd.Index = (
            pd.RangeIndex(n) if current_dates is None else pd.Index(current_dates)
        )
        self.horizon = int(future_dates)
        self.test_length = int(test_length)
        self.exog = exog.to_numpy(dtype=float) if exog is not None else None
        self.exog_names = list(exog.columns) if exog is not None else []
        self.name = name
        self.seed = seed
        self.device = resolve_device(device)

        self.transformer: Transformer | None = None
        self.reverter: Reverter | None = None
        self.retriever: AnalogRetriever | None = None
        self.spec = ModelSpec()
        self.result: ForecastResult | None = None
        self._last_trainers: list[Trainer] | None = None
        self._saved_residuals: np.ndarray | None = None
        self._saved_alpha: float = 0.1

        if n < self.horizon + self.test_length + self.spec.lags:
            # Not fatal here (spec may change), but warn the caller early via attribute.
            self._too_short = True
        else:
            self._too_short = False

    # ------------------------------------------------------------------ helpers
    def attach_transformer(self, transformer: Transformer, reverter: Reverter | None = None) -> None:
        self.transformer = transformer
        self.reverter = reverter or transformer.reverter()

    def attach_retriever(self, retriever: AnalogRetriever) -> None:
        """Attach a RAG analog retriever to add retrieved-continuation feature channels."""
        self.retriever = retriever

    def _future_index(self, n_steps: int) -> pd.DatetimeIndex:
        if isinstance(self.dates, pd.DatetimeIndex):
            freq = pd.infer_freq(self.dates) or "B"
            start = self.dates[-1]
            return pd.date_range(start=start, periods=n_steps + 1, freq=freq)[1:]
        return pd.RangeIndex(len(self.dates), len(self.dates) + n_steps)  # type: ignore[return-value]

    def _build_feature_matrix(
        self,
        target_transformed: np.ndarray,
        exog_slice: np.ndarray | None,
        *,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Assemble ``(T, F)`` features: transformed target, standardized exog, RAG channels."""
        cols = [target_transformed.reshape(-1, 1)]
        if exog_slice is not None and exog_slice.shape[1] > 0:
            mean = exog_slice.mean(axis=0, keepdims=True)
            std = exog_slice.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            cols.append((exog_slice - mean) / std)
        if self.retriever is not None:
            analog = self.retriever.feature_channels(target_transformed)
            if analog is not None and analog.shape[1] > 0:
                cols.append(analog)
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _train_models(
        self,
        target: np.ndarray,
        exog_slice: np.ndarray | None,
        horizon: int,
    ) -> tuple[list[Trainer], np.ndarray]:
        """Fit ``spec.ensemble`` differently-seeded models and return ``(trainers, window)``.

        ``target`` is in the *original* scale; the (already-fit) transformer is applied here.
        Averaging an ensemble of models reduces variance from random initialisation and
        gives better-calibrated point forecasts. The returned window is for one-shot
        future prediction.
        """
        positions = np.arange(len(target))
        if self.transformer is not None:
            t_target = self.transformer.transform(target, positions)
        else:
            t_target = target
        features = self._build_feature_matrix(t_target, exog_slice, positions=positions)

        x, y = make_windows(features, lags=self.spec.lags, horizon=horizon, target_idx=0)
        if self.spec.target_mode == "delta":
            # Predict the change from the last input value, so the model learns deviations
            # from a random-walk (naive) baseline rather than the non-stationary level.
            anchors = x[:, -1, 0:1]  # (n, 1) last target value of each input window
            y = y - anchors

        trainers: list[Trainer] = []
        for member in range(max(1, self.spec.ensemble)):
            set_seed(self.seed + member)  # de-correlate ensemble members
            model = LSTMForecaster(
                n_features=features.shape[1],
                horizon=horizon,
                hidden_size=self.spec.hidden_size,
                num_layers=self.spec.num_layers,
                dropout=self.spec.dropout,
                use_attention=self.spec.use_attention,
                quantiles=self.spec.quantiles,
            )
            cfg = TrainerConfig(
                epochs=self.spec.epochs,
                batch_size=self.spec.batch_size,
                lr=self.spec.lr,
                weight_decay=self.spec.weight_decay,
                patience=self.spec.patience,
                val_fraction=self.spec.val_fraction,
                device=self.device,
                quantiles=self.spec.quantiles,
            )
            trainers.append(Trainer(model, cfg).fit(x, y))
        window = last_input_window(features, lags=self.spec.lags)
        return trainers, window

    def _predict_horizon(
        self,
        trainers: list[Trainer],
        window: np.ndarray,
        horizon: int,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Ensemble-average a horizon forecast and revert to the original scale."""
        preds = np.stack([t.predict_point(window).ravel()[:horizon] for t in trainers], axis=0)
        pred_t = preds.mean(axis=0)
        if self.spec.target_mode == "delta":
            # Add back the anchor (last transformed target value of the input window).
            pred_t = pred_t + float(window[0, -1, 0])
        if self.transformer is not None:
            return self.transformer.inverse_transform(pred_t, positions)
        return pred_t

    # ------------------------------------------------------------------- public
    def fit_predict(
        self,
        spec: ModelSpec | None = None,
        *,
        alpha: float = 0.1,
        run_backtest: bool = False,
        backtest_windows: int = 10,
        benchmark: bool = True,
    ) -> ForecastResult:
        """Run the full evaluation + production forecast pipeline."""
        if spec is not None:
            self.spec = spec
        set_seed(self.seed)
        n = self.y.size
        lags = self.spec.lags

        if n < self.test_length + lags + 1:
            raise ValueError(
                f"Series too short ({n}) for test_length={self.test_length} and lags={lags}."
            )

        # ---- 1. Test-set evaluation (transformer fit on train portion only) -------------
        split = n - self.test_length
        train_target = self.y[:split]
        train_exog = self.exog[:split] if self.exog is not None else None
        test_actual = self.y[split:]

        if self.transformer is not None:
            self.transformer.fit(train_target, np.arange(split))
        trainers, window = self._train_models(train_target, train_exog, self.test_length)
        test_positions = np.arange(split, n)
        test_pred = self._predict_horizon(trainers, window, self.test_length, test_positions)

        # The model's own test metric is always computed (CV/tuning relies on it); baselines
        # and the significance test are only run when benchmarking.
        metrics: dict[str, dict[str, float]] = {
            self.name: point_metrics(
                test_actual, test_pred, y_train=train_target, season=max(lags, 1)
            )
        }
        significance: dict[str, object] = {}
        if benchmark:
            baseline_preds: dict[str, np.ndarray] = {}
            for bname, model in baseline_registry(season=min(5, lags)).items():
                bpred = model.fit(train_target).predict(self.test_length)
                baseline_preds[bname] = bpred
                metrics[bname] = point_metrics(
                    test_actual, bpred, y_train=train_target, season=max(lags, 1)
                )
            # Diebold-Mariano: is the model significantly better than naive?
            if "naive" in baseline_preds:
                dm = diebold_mariano(
                    test_actual - test_pred, test_actual - baseline_preds["naive"]
                )
                verdict = {"a": "model", "b": "naive", "tie": "no significant difference"}[dm.better]
                significance["vs_naive"] = {**dm.as_dict(), "winner": verdict}

        # Residuals on the test window double as the conformal calibration set.
        test_residuals = test_actual - test_pred

        # ---- 2. Production forecast (transformer refit on all data) ----------------------
        if self.transformer is not None:
            self.transformer.fit(self.y, np.arange(n))
        final_trainers, final_window = self._train_models(self.y, self.exog, self.horizon)
        self._last_trainers = final_trainers
        future_positions = np.arange(n, n + self.horizon)
        point = self._predict_horizon(final_trainers, final_window, self.horizon, future_positions)

        # ---- 3. Intervals ---------------------------------------------------------------
        bt_result: BacktestResult | None = None
        if run_backtest:
            bt_result = self._run_backtest(alpha=alpha, n_windows=backtest_windows)
            lower, upper = dynamic_intervals(point, bt_result.residual_matrix, alpha=alpha)
        else:
            lower, upper = conformal_intervals(point, test_residuals, alpha=alpha)

        future_dates = self._future_index(self.horizon)
        test_dates = self.dates[split:] if isinstance(self.dates, pd.DatetimeIndex) else None

        ivl = (
            interval_metrics(test_actual, *conformal_intervals(test_pred, test_residuals, alpha),
                             nominal=1 - alpha)
            if benchmark
            else {}
        )

        self.result = ForecastResult(
            point=point,
            lower=lower,
            upper=upper,
            future_dates=future_dates,
            history_dates=self.dates,  # type: ignore[arg-type]
            history_values=self.y,
            test_dates=test_dates,  # type: ignore[arg-type]
            test_actual=test_actual,
            test_pred=test_pred,
            metrics=metrics,
            interval=ivl,
            significance=significance,
            backtest_result=bt_result,
            alpha=alpha,
        )
        return self.result

    def tune(
        self,
        specs: list[ModelSpec],
        *,
        k: int = 3,
        val_length: int | None = None,
        cv_epochs: int = 40,
    ) -> dict[str, object]:
        """Walk-forward cross-validate ``specs`` and adopt the best one.

        Sets ``self.spec`` to the lowest mean-CV-RMSE candidate and returns a report. Pair
        with :func:`lstm_forecast.forecasting.tuning.specs_from_suggestion` to evaluate an
        AI-proposed grid — the LLM narrows the search; CV makes the final call.
        """
        from lstm_forecast.forecasting.tuning import (
            _transformer_factory_from,
            walk_forward_cv,
        )

        factory = _transformer_factory_from(self.transformer)
        results: list[dict[str, object]] = []
        best_spec: ModelSpec | None = None
        best_score = float("inf")
        for spec in specs:
            score = walk_forward_cv(
                self.y,
                spec=spec,
                exog=self.exog,
                transformer_factory=factory,
                retriever=self.retriever,
                k=k,
                val_length=val_length,
                cv_epochs=cv_epochs,
                seed=self.seed,
                device=self.device,
            )
            results.append(
                {"lags": spec.lags, "hidden_size": spec.hidden_size,
                 "num_layers": spec.num_layers, "dropout": spec.dropout, "cv_rmse": score}
            )
            if score < best_score:
                best_score, best_spec = score, spec
        if best_spec is not None:
            self.spec = best_spec
        return {"best_spec": best_spec, "best_cv_rmse": best_score, "results": results}

    def _run_backtest(self, *, alpha: float, n_windows: int) -> BacktestResult:
        """Backtest the LSTM by refitting at multiple cutoffs (expensive)."""
        # Use a lighter spec during backtesting (fewer epochs, single model) to bound cost.
        original_epochs = self.spec.epochs
        original_ensemble = self.spec.ensemble
        self.spec.epochs = min(original_epochs, 40)
        self.spec.ensemble = 1

        def fit_predict_fn(train: np.ndarray) -> np.ndarray:
            positions = np.arange(len(train))
            if self.transformer is not None:
                self.transformer.fit(train, positions)
            trainers, window = self._train_models(train, None, self.horizon)
            fut_pos = np.arange(len(train), len(train) + self.horizon)
            return self._predict_horizon(trainers, window, self.horizon, fut_pos)

        try:
            return backtest(
                fit_predict_fn, self.y, horizon=self.horizon, n_windows=n_windows, step=max(1, self.horizon // 2)
            )
        finally:
            self.spec.epochs = original_epochs
            self.spec.ensemble = original_ensemble
            # Restore the transformer fit on the full series for downstream reverts.
            if self.transformer is not None:
                self.transformer.fit(self.y, np.arange(self.y.size))

    def transfer_predict(
        self,
        *,
        transfer_from: Forecaster,
        future_dates: int | None = None,
    ) -> ForecastResult:
        """Forecast this (new) series using a model trained on ``transfer_from``.

        Implements the design doc's transfer-learning scenarios: apply a model fit on one
        series to new data from the same series, or to a different but similar series — no
        retraining. Requires ``transfer_from`` to have been fit (``fit_predict`` called)
        and to share this Forecaster's lag/feature configuration.
        """
        if not transfer_from._last_trainers:
            raise RuntimeError("transfer_from must be fit (call fit_predict first).")
        set_seed(self.seed)
        self.spec = transfer_from.spec
        self.transformer = transfer_from.transformer
        self.reverter = transfer_from.reverter
        self.retriever = transfer_from.retriever

        h = int(future_dates or self.horizon)
        n = self.y.size
        positions = np.arange(n)
        if self.transformer is not None:
            t_target = self.transformer.transform(self.y, positions)
        else:
            t_target = self.y
        features = self._build_feature_matrix(t_target, self.exog, positions=positions)
        window = last_input_window(features, lags=self.spec.lags)
        fut_pos = np.arange(n, n + h)
        point = self._predict_horizon(transfer_from._last_trainers, window, h, fut_pos)

        # Reuse the source model's calibration residuals if available.
        if transfer_from.result is not None and transfer_from.result.test_actual is not None:
            res = transfer_from.result.test_actual - transfer_from.result.test_pred  # type: ignore[operator]
            radius = conformal_quantile(res, alpha=transfer_from.result.alpha)
        else:
            radius = float(np.std(self.y) * 1.64)
        lower, upper = point - radius, point + radius

        self.result = ForecastResult(
            point=point,
            lower=lower,
            upper=upper,
            future_dates=self._future_index(h),
            history_dates=self.dates,  # type: ignore[arg-type]
            history_values=self.y,
            alpha=transfer_from.result.alpha if transfer_from.result else 0.1,
        )
        return self.result

    # ----------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Persist the fitted production model so forecasts need no retraining.

        Saves the ensemble weights, the fitted transformer, the spec, the calibration
        residuals (for intervals) and the series/exog history. Load with
        :meth:`Forecaster.load` and call :meth:`forecast_future`. Requires ``fit_predict``.
        """
        import torch

        if not self._last_trainers:
            raise RuntimeError("Call fit_predict before save().")
        residuals = None
        alpha = 0.1
        if self.result is not None and self.result.test_actual is not None:
            residuals = self.result.test_actual - self.result.test_pred
            alpha = self.result.alpha
        models = [
            {
                "state_dict": tr.model.state_dict(),
                "ctor": {
                    "n_features": tr.model.n_features,
                    "horizon": tr.model.horizon,
                    "hidden_size": self.spec.hidden_size,
                    "num_layers": self.spec.num_layers,
                    "dropout": self.spec.dropout,
                    "use_attention": tr.model.use_attention,
                    "quantiles": tr.model.quantiles,
                },
            }
            for tr in self._last_trainers
        ]
        payload = {
            "format_version": 1,
            "spec": asdict(self.spec),
            "name": self.name,
            "seed": self.seed,
            "horizon": self.horizon,
            "test_length": self.test_length,
            "y": self.y,
            "exog": self.exog,
            "exog_names": self.exog_names,
            "dates": self.dates,
            "transformer": self.transformer,
            "retriever": self.retriever,
            "residuals": residuals,
            "alpha": alpha,
            "models": models,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> Forecaster:
        """Load a model saved with :meth:`save`, ready for :meth:`forecast_future`."""
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        exog_df = (
            pd.DataFrame(payload["exog"], columns=payload["exog_names"])
            if payload["exog"] is not None
            else None
        )
        f = cls(
            y=payload["y"],
            current_dates=payload["dates"],
            future_dates=payload["horizon"],
            test_length=payload["test_length"],
            exog=exog_df,
            name=payload["name"],
            seed=payload["seed"],
            device=device,
        )
        f.spec = ModelSpec(**payload["spec"])
        f.transformer = payload["transformer"]
        f.reverter = f.transformer.reverter() if f.transformer is not None else None
        f.retriever = payload["retriever"]
        f._saved_residuals = payload["residuals"]
        f._saved_alpha = payload["alpha"]

        trainers: list[Trainer] = []
        for md in payload["models"]:
            model = LSTMForecaster(**md["ctor"])
            model.load_state_dict(md["state_dict"])
            cfg = TrainerConfig(device=f.device, quantiles=f.spec.quantiles)
            trainers.append(Trainer(model, cfg))
        f._last_trainers = trainers
        return f

    def forecast_future(self, *, alpha: float | None = None) -> ForecastResult:
        """Produce a future forecast from already-fitted models (no training).

        Works after :meth:`fit_predict` or :meth:`load`. Intervals reuse the stored
        calibration residuals when available, else a normal-approximation fallback.
        """
        if not self._last_trainers:
            raise RuntimeError("No fitted model; call fit_predict or load first.")
        n = self.y.size
        positions = np.arange(n)
        t_target = (
            self.transformer.transform(self.y, positions) if self.transformer is not None
            else self.y
        )
        features = self._build_feature_matrix(t_target, self.exog, positions=positions)
        window = last_input_window(features, lags=self.spec.lags)
        fut_pos = np.arange(n, n + self.horizon)
        point = self._predict_horizon(self._last_trainers, window, self.horizon, fut_pos)

        a = alpha if alpha is not None else getattr(self, "_saved_alpha", 0.1)
        residuals = getattr(self, "_saved_residuals", None)
        if residuals is not None and len(residuals):
            lower, upper = conformal_intervals(point, residuals, alpha=a)
        else:
            radius = float(np.std(self.y) * 1.64)
            lower, upper = point - radius, point + radius

        self.result = ForecastResult(
            point=point,
            lower=lower,
            upper=upper,
            future_dates=self._future_index(self.horizon),
            history_dates=self.dates,
            history_values=self.y,
            alpha=a,
        )
        return self.result

    def plot(self, ci: bool = True, ax: Any = None, show_test: bool = True):
        """Plot history, forecast and (optionally) intervals and the test set."""
        import matplotlib.pyplot as plt

        if self.result is None:
            raise RuntimeError("Nothing to plot — call fit_predict first.")
        r = self.result
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 5))
        ax.plot(r.history_dates, r.history_values, label="history", color="#1f4e79")
        ax.plot(r.future_dates, r.point, label="forecast", color="#c0392b")
        if ci:
            ax.fill_between(r.future_dates, r.lower, r.upper, color="#c0392b", alpha=0.2,
                            label=f"{int((1 - r.alpha) * 100)}% interval")
        if show_test and r.test_dates is not None and r.test_pred is not None:
            ax.plot(r.test_dates, r.test_pred, "--", color="#e67e22", label="test forecast")
        ax.set_title(f"{self.name} forecast")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        return ax
