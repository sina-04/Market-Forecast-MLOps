import numpy as np
import pandas as pd
import pytest

from marketforecast.data import clean, adjust_prices
from marketforecast.features import engineer


@pytest.fixture
def raw_frame():
    count = 420
    i = np.arange(count)
    # Deterministic toy data solely for testing; never presented as market evidence.
    close = 1000 + 0.7 * i + 12 * np.sin(i / 9)
    return pd.DataFrame({
        "date": pd.bdate_range("2022-01-01", periods=count).strftime("%Y%m%d"),
        "open": close - 1, "high": close + 8, "low": close - 8,
        "official_close": close, "last_trade": close + 1,
        "previous_official_close": np.r_[close[0], close[:-1]],
        "volume": (10000 + i * 20).astype(int), "value": close * 10000,
        "trade_count": np.full(count, 30), "instrument_id": "123",
    })


@pytest.fixture
def feature_frame(raw_frame):
    cleaned, _, _ = clean(raw_frame)
    adjusted = adjust_prices(cleaned)
    adjusted["session_number"] = range(len(adjusted))
    return engineer(adjusted)


@pytest.fixture
def config():
    from marketforecast.io import read_json
    return read_json("configs/default.json")

