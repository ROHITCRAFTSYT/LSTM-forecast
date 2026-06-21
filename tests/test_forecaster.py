"""Integration tests for the public Forecaster and Pipeline."""

from __future__ import annotations

import numpy as np

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.forecasting.forecaster import ModelSpec
from lstm_forecast.transforms import default_finance_transformer


def _fast_spec(**kw):
    base = {"lags": 12, "hidden_size": 16, "epochs": 15, "patience": 5}
    base.update(kw)
    return ModelSpec(**base)


def test_fit_predict_shapes_and_metrics(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=8, test_length=15)
    res = f.fit_predict(_fast_spec())
    assert res.point.shape == (8,)
    assert res.lower.shape == (8,) and res.upper.shape == (8,)
    assert (res.upper >= res.lower).all()
    assert len(res.future_dates) == 8
    # Benchmark table includes the model and baselines.
    assert "lstm" in res.metrics and "naive" in res.metrics
    frame = res.metrics_frame()
    assert "rmse" in frame.columns


def test_intervals_contain_point(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=6, test_length=12)
    res = f.fit_predict(_fast_spec())
    assert (res.lower <= res.point).all()
    assert (res.upper >= res.point).all()


def test_pipeline_with_transformer(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=6, test_length=12)
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    res = pipe.fit_predict(f, spec=_fast_spec())
    assert res.point.shape == (6,)
    # Forecast should be in the original price scale (same order of magnitude as history).
    assert 0.3 * prices["close"].mean() < res.point.mean() < 3 * prices["close"].mean()


def test_delta_mode_anchors_near_last_value(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=5, test_length=10)
    res = f.fit_predict(_fast_spec(target_mode="delta"))
    last = float(prices["close"].iloc[-1])
    # First forecast step should be within a few percent of the last observed value.
    assert abs(res.point[0] - last) / last < 0.15


def test_backtest_dynamic_intervals(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=6, test_length=12)
    res = f.fit_predict(_fast_spec(epochs=8), run_backtest=True, backtest_windows=4)
    assert res.backtest_result is not None
    assert res.backtest_result.residual_matrix.shape[1] == 6


def test_transfer_predict(prices):
    src = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=6, test_length=12)
    src.fit_predict(_fast_spec())
    other = prices["close"] * 1.1 + 2.0
    tgt = Forecaster(y=other, current_dates=prices.index, future_dates=6, test_length=12)
    res = tgt.transfer_predict(transfer_from=src)
    assert res.point.shape == (6,)
    assert (res.upper >= res.lower).all()


def test_rag_pipeline_runs(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=6, test_length=12)
    transformer, _ = default_finance_transformer(seasonal_period=5)
    split = f.y.size - f.test_length
    transformer.fit(f.y[:split], np.arange(split))
    ref = transformer.transform(f.y[:split], np.arange(split))
    from lstm_forecast.rag import build_analog_retriever

    f.attach_transformer(transformer)
    f.attach_retriever(build_analog_retriever(ref, window_len=12, k=5))
    res = f.fit_predict(_fast_spec())
    assert res.point.shape == (6,)


def test_ensemble_trains_multiple_members(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=5, test_length=10)
    res = f.fit_predict(_fast_spec(ensemble=3))
    assert f._last_trainers is not None and len(f._last_trainers) == 3
    assert res.point.shape == (5,)


def test_significance_vs_naive_present(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=6, test_length=15)
    res = f.fit_predict(_fast_spec())
    assert "vs_naive" in res.significance
    dm = res.significance["vs_naive"]
    assert "p_value" in dm and "winner" in dm and "significant" in dm


def test_save_load_roundtrip_no_retrain(prices, tmp_path):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=6, test_length=12)
    res = f.fit_predict(_fast_spec())
    path = tmp_path / "model.pt"
    f.save(path)
    assert path.exists()

    loaded = Forecaster.load(path)
    out = loaded.forecast_future()
    # Reloaded model reproduces the original forecast without retraining.
    assert out.point.shape == res.point.shape
    np.testing.assert_allclose(out.point, res.point, rtol=1e-4, atol=1e-3)
    assert (out.upper >= out.lower).all()


def test_result_to_dict_serialisable(prices):
    f = Forecaster(y=prices["close"], current_dates=prices.index, future_dates=4, test_length=8)
    res = f.fit_predict(_fast_spec())
    import json

    d = res.to_dict()
    json.dumps(d)  # must not raise
    assert len(d["point"]) == 4
