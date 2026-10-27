from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from marketforecast.data import adjust_prices
from marketforecast.import_export import import_exports, MAPPING
from marketforecast.workflow import prepare_run


def exports(tmp_path, raw_frame):
    raw = raw_frame.drop(columns="instrument_id").copy()
    raw.date = pd.date_range("2021-09-26", "2026-09-26", periods=len(raw)).strftime("%Y-%m-%d")
    raw.loc[250, "open"] = 0
    adjusted = adjust_prices(raw)
    other = raw.copy()
    for name in ["open", "high", "low", "official_close", "last_trade"]:
        other[name] = adjusted["adj_" + name].round()
    paths = [tmp_path / "raw.csv", tmp_path / "adjusted.csv"]
    for frame, path in zip([raw, other], paths):
        frame.rename(columns={v:k for k,v in MAPPING.items()}).to_csv(path,index=False)
    return paths


def test_import_preserves_chain_and_excludes_windows_across_invalid_open(tmp_path, raw_frame, config):
    paths = exports(tmp_path, raw_frame)
    snapshot, manifest = import_exports(tmp_path / "run", config, *paths,
        now=datetime(2026,9,27,tzinfo=timezone.utc))
    assert (snapshot / "source_unadjusted.csv").read_bytes() == paths[0].read_bytes()
    assert manifest["end_date"] == "2026-09-26"
    _, frame, _, plan, _, audits = prepare_run(tmp_path / "run", config)
    assert audits["cleaning"]["missing_open_sessions_excluded"] == 1
    assert 250 not in frame.session_number.to_numpy()
    for t in plan.target_positions:
        assert np.all(np.diff(frame.session_number.iloc[t-60:t+1]) == 1)
    # Removing the candle must not fabricate a corporate-action coefficient.
    original = pd.read_csv(snapshot / "raw.csv")
    expected = adjust_prices(original).set_index("date")
    np.testing.assert_allclose(frame.adj_official_close,
        expected.loc[frame.date.dt.strftime("%Y-%m-%d"),"adj_official_close"])


def test_import_rejects_wrong_adjustment(tmp_path, raw_frame, config):
    paths = exports(tmp_path, raw_frame)
    adjusted = pd.read_csv(paths[1]); adjusted.loc[10,"adjClose"] += 5
    adjusted.to_csv(paths[1],index=False)
    with pytest.raises(ValueError,match="disagrees"):
        import_exports(tmp_path / "run", config, *paths)
