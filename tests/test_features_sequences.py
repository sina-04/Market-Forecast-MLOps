import numpy as np
import pandas as pd

from marketforecast.data import adjust_prices, clean
from marketforecast.evaluation import metrics
from marketforecast.features import engineer, wilder_average
from marketforecast.sequences import make_plan, prepare


def test_features_causal_on_fixed_adjusted_basis(raw_frame):
    cleaned, _, _ = clean(raw_frame)
    fixed = adjust_prices(cleaned)
    first, names, _ = engineer(fixed)
    changed = fixed.copy()
    cutoff = 250
    changed.loc[cutoff:, [c for c in changed if c.startswith("adj_")]] *= 2
    changed.loc[cutoff:, "volume"] *= 3
    second, _, _ = engineer(changed)
    mask = first["date"] < fixed.loc[cutoff, "date"]
    pd.testing.assert_frame_equal(first.loc[mask, names], second.loc[mask, names])


def test_flat_prices_rsi_and_bollinger_are_finite(raw_frame):
    for column in ["open", "high", "low", "official_close", "last_trade", "previous_official_close"]:
        raw_frame[column] = 1000
    cleaned, _, _ = clean(raw_frame)
    features, names, _ = engineer(adjust_prices(cleaned))
    assert (features["rsi_14"] == 50).all()
    assert (features["bollinger_position"] == 0).all()
    assert np.isfinite(features[names]).all().all()


def test_wilder_seed():
    result = wilder_average(pd.Series([1., 2., 3., 4.]), period=3)
    assert np.isnan(result.iloc[0])
    assert result.iloc[2] == 2
    assert result.iloc[3] == (2 * 2 + 4) / 3


def test_alignment_common_dates_and_train_scalers(feature_frame, config):
    frame, names, _ = feature_frame
    plan = make_plan(frame, config)
    dates = []
    for length in [10, 30, 60]:
        splits, xs, ys = prepare(frame, names, length, plan)
        dates.append(splits["test"]["dates"])
        train = splits["train"]
        assert train["dates"][-1] < splits["validation"]["dates"][0] < splits["test"]["dates"][0]
        t = train["target_positions"][0]
        np.testing.assert_allclose(train["X"][0], xs.transform(frame.iloc[t-length:t][names].to_numpy()), rtol=1e-5, atol=1e-5)
        assert train["actual"][0] == frame.iloc[t]["adj_official_close"]
        assert train["persistence"][0] == frame.iloc[t-1]["adj_official_close"]
        np.testing.assert_allclose(ys.mean_, train["actual"].mean())
        indices = np.unique(np.concatenate([np.arange(t-length, t) for t in train["target_positions"]]))
        np.testing.assert_allclose(xs.mean_, frame.iloc[indices][names].mean())
    for value in dates[1:]:
        np.testing.assert_array_equal(value, dates[0])


def test_future_extreme_values_do_not_change_training_scalers(feature_frame, config):
    frame, names, _ = feature_frame
    plan = make_plan(frame, config)
    splits, xs, ys = prepare(frame, names, 30, plan)
    changed = frame.copy()
    start = splits["validation"]["target_positions"][0]
    changed.loc[start:, names] *= 100000
    _, future_xs, future_ys = prepare(changed, names, 30, plan)
    np.testing.assert_allclose(xs.mean_, future_xs.mean_)
    np.testing.assert_allclose(ys.mean_, future_ys.mean_)


def test_missing_feature_rows_do_not_shift_horizon(feature_frame, config):
    frame, _, _ = feature_frame
    frame = frame.drop(index=150).reset_index(drop=True)
    plan = make_plan(frame, config)
    for target in plan.target_positions:
        assert np.all(np.diff(frame.iloc[target-60:target+1]["session_number"]) == 1)


def test_original_unit_metrics():
    result = metrics([100, 200], [110, 180])
    assert result["MAE"] == 15
    assert result["MSE"] == 250
    assert result["MAPE"] == 10
    assert metrics([0, 100], [1, 110])["MAPE"] == 10
