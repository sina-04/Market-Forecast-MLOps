from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from marketforecast.hourly import clean_hourly, import_hourly
from marketforecast.features import engineer, ohlcv_features
from marketforecast.io import read_json
from marketforecast.sequences import make_plan, prepare
from marketforecast.training import restore_prices
from marketforecast.workflow import prepare_run


def hourly_fixture(tmp_path, config):
    days = pd.bdate_range("2022-01-01", periods=110)
    close = np.linspace(1000, 1200, len(days))
    daily = pd.DataFrame({"date": days, "open": close, "high": close + 10, "low": close - 10,
                          "adjClose": close, "close": close, "yesterday": np.r_[close[0], close[:-1]],
                          "count": 40, "volume": 4000, "value": close * 4000})
    # A daily corporate action must apply the same factor to all four bars, not four times.
    daily.loc[50, "yesterday"] *= .5
    stamps = [day + pd.Timedelta(hours=h) for day in days for h in range(9, 13) if day != days[70]]
    i = np.arange(len(stamps))
    price = 1000 + i*.3 + 4*np.sin(i/10)
    bars = pd.DataFrame({"datetime": stamps, "open": price, "high": price+2, "low": price-2,
                         "hourly_last_price": price, "volume": 1000+i, "returned_trade_count": 10})
    coverage = pd.DataFrame({"date": days, "status": ["unresolved" if d == days[70] else "downloaded" for d in days],
                             "hourly_bars": [0 if d == days[70] else 4 for d in days],
                             "returned_trades": 40, "error": ""})
    paths = [tmp_path/name for name in ["bars.csv", "daily.csv", "coverage.csv"]]
    for frame, path in zip([bars, daily, coverage], paths):
        frame.to_csv(path, index=False)
    hourly_config = {**config, "frequency": "hourly", "target_mode": "log_return"}
    snapshot, _ = import_hourly(tmp_path/"experiment", hourly_config, *paths,
                                now=datetime(2026, 10, 5, tzinfo=timezone.utc))
    return snapshot, hourly_config


def test_hourly_adjustment_timestamps_gaps_and_scalers(tmp_path, config):
    snapshot, config = hourly_fixture(tmp_path, config)
    output, frame, names, plan, manifest, audits = prepare_run(tmp_path/"experiment", config)
    raw = pd.read_csv(snapshot/"raw.csv")
    cleaned, _, _ = clean_hourly(raw, snapshot)
    assert cleaned.groupby("session_date").adjustment_factor.nunique().eq(1).all()
    assert cleaned.iloc[0].adjustment_factor == pytest.approx(.5)
    assert cleaned.iloc[-1].adjustment_factor == 1
    assert audits["cleaning"]["unresolved_dates"] == 1
    assert {"hour_sin", "hour_cos", "elapsed_hours_feature"}.issubset(names)
    assert set(ohlcv_features(frame)).issubset(names)
    splits, xs, ys = prepare(frame, names, 30, plan)
    assert len(set(splits["test"]["dates"])) == len(splits["test"]["dates"])
    assert all(len(d) == 19 for d in splits["test"]["dates"])
    for target in plan.target_positions:
        assert np.diff(frame.session_number.iloc[target-60:target+1]).min() == 1
        assert frame.segment.iloc[target-60:target+1].nunique() == 1
    labels = np.log(frame.adj_hourly_last_price / frame.adj_hourly_last_price.shift())
    np.testing.assert_allclose(ys.mean_[0], labels.iloc[plan.target_positions[:plan.train_end]].mean())
    restored = restore_prices(splits["test"]["y"], ys, splits["test"]["persistence"], "log_return")
    np.testing.assert_allclose(restored, splits["test"]["actual"], rtol=1e-6)
    changed = cleaned.copy()
    cutoff = 200
    changed.loc[cutoff:, [c for c in changed if c.startswith("adj_")]] *= 3
    changed.loc[cutoff:, "volume"] *= 2
    other, _, _ = engineer(changed)
    pd.testing.assert_frame_equal(frame.loc[frame.date < cleaned.date.iloc[cutoff], names].reset_index(drop=True),
                                  other.loc[other.date < cleaned.date.iloc[cutoff], names].reset_index(drop=True))
    assert read_json(output/"snapshot_manifest.json")["frequency"] == "hourly"


def test_hourly_source_cannot_be_used_as_daily(tmp_path, config):
    hourly_fixture(tmp_path, config)
    with pytest.raises(ValueError, match="Config disagrees"):
        prepare_run(tmp_path/"experiment", config)


def test_zero_log_return_reconstructs_persistence():
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit([[-.1], [.1]])
    np.testing.assert_allclose(restore_prices([[0], [0]], scaler, [100, 200], "log_return"), [100, 200])
