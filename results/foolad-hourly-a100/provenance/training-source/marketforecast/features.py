"""Causal indicators calculated on an already fixed adjustment basis."""
import numpy as np
import pandas as pd

OHLCV_FEATURES = ["adj_open", "adj_high", "adj_low", "adj_official_close", "volume"]


def price_column(frame):
    return "adj_hourly_last_price" if "adj_hourly_last_price" in frame else "adj_official_close"


def ohlcv_features(frame):
    if "adj_hourly_last_price" in frame:
        return ["open_to_last", "high_to_last", "low_to_last", "return_1", "log_volume"]
    return OHLCV_FEATURES.copy()


def safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def wilder_average(series, period=14):
    """Wilder smoothing, seeded with the first period's simple average."""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    valid_start = next((i for i in range(len(values) - period + 1)
                        if np.isfinite(values[i:i + period]).all()), None)
    if valid_start is None:
        return pd.Series(result, index=series.index)
    seed = valid_start + period - 1
    result[seed] = values[valid_start:seed + 1].mean()
    for i in range(seed + 1, len(values)):
        result[i] = (result[i - 1] * (period - 1) + values[i]) / period
    return pd.Series(result, index=series.index)


def engineer(cleaned):
    if "adj_hourly_last_price" in cleaned and "segment" in cleaned:
        pieces = [_engineer(group)[0] for _, group in cleaned.groupby("segment", sort=True)]
        output = pd.concat(pieces, ignore_index=True)
        _, names, _ = _engineer(cleaned.iloc[:0])
        return output, names, {"input_rows": len(cleaned), "usable_rows": len(output),
                               "warmup_or_nonfinite_rows": len(cleaned) - len(output),
                               "indicator_policy": "Restart indicators after unresolved trading dates; periods count observed bars"}
    return _engineer(cleaned)


def _engineer(cleaned):
    frame = cleaned.copy()
    close, high, low = (frame[c] for c in [price_column(frame), "adj_high", "adj_low"])
    volume = frame["volume"]
    features = OHLCV_FEATURES.copy()

    def add(name, values):
        frame[name] = values
        features.append(name)

    returns = close.pct_change(fill_method=None)
    add("return_1", returns)
    for lag in [1, 2, 5]:
        add(f"close_lag_{lag}", close.shift(lag))
        add(f"return_lag_{lag}", returns.shift(lag))
    for window in [5, 10, 20]:
        sma = close.rolling(window).mean()
        add(f"sma_{window}", sma)
        add(f"ema_{window}", close.ewm(span=window, adjust=False, min_periods=window).mean())
        add(f"volatility_{window}", returns.rolling(window).std())
        add(f"rolling_max_{window}", close.rolling(window).max())
        add(f"rolling_min_{window}", close.rolling(window).min())
        add(f"price_to_sma_{window}", safe_divide(close, sma) - 1)
    delta = close.diff()
    gain = wilder_average(delta.clip(lower=0))
    loss = wilder_average(-delta.clip(upper=0))
    rsi = 100 - 100 / (1 + safe_divide(gain, loss))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
    add("rsi_14", rsi)
    macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    add("macd", macd)
    add("macd_signal", signal)
    add("macd_histogram", macd - signal)
    mean, std = close.rolling(20).mean(), close.rolling(20).std()
    add("bollinger_upper", mean + 2 * std)
    add("bollinger_lower", mean - 2 * std)
    add("bollinger_position", safe_divide(close - mean, 2 * std).fillna(0).where(mean.notna()))
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    add("atr_14", wilder_average(true_range))
    add("volume_change", volume.pct_change(fill_method=None))
    add("relative_volume_20", safe_divide(volume, volume.rolling(20).mean()))
    add("range_to_close", safe_divide(high - low, close))
    add("return_times_volume", returns * volume)
    add("elapsed_days_feature", frame["elapsed_days"])
    dates = pd.to_datetime(frame["date"])
    add("weekday_sin", np.sin(2 * np.pi * dates.dt.dayofweek / 7))
    add("weekday_cos", np.cos(2 * np.pi * dates.dt.dayofweek / 7))
    month = frame["jalali_date"].str.slice(5, 7).astype(int)
    add("jalali_month_sin", np.sin(2 * np.pi * month / 12))
    add("jalali_month_cos", np.cos(2 * np.pi * month / 12))
    # Calendar month-end is knowable at forecast time; last observed monthly session is not.
    add("gregorian_month_end", dates.dt.is_month_end.astype(int))
    if "adj_hourly_last_price" in frame:
        # Relative inputs avoid learning the historical price scale across corporate actions/regimes.
        for name in ["adj_open", "adj_high", "adj_low", "adj_official_close", "volume"]:
            if name in features:
                features.remove(name)
        for name in list(features):
            if name.startswith(("close_lag_", "sma_", "ema_", "rolling_max_", "rolling_min_")):
                frame[name] = safe_divide(frame[name], close) - 1
            elif name in ["macd", "macd_signal", "macd_histogram", "bollinger_upper", "bollinger_lower", "atr_14"]:
                frame[name] = safe_divide(frame[name], close)
        features.remove("return_times_volume")
        add("open_to_last", safe_divide(frame.adj_open, close) - 1)
        add("high_to_last", safe_divide(high, close) - 1)
        add("low_to_last", safe_divide(low, close) - 1)
        add("log_volume", np.log1p(volume))
        add("hour_sin", np.sin(2 * np.pi * dates.dt.hour / 24))
        add("hour_cos", np.cos(2 * np.pi * dates.dt.hour / 24))
        add("elapsed_hours_feature", frame["elapsed_hours"])
    finite = np.isfinite(frame[features]).all(axis=1)
    output = frame.loc[finite].reset_index(drop=True)
    return output, features, {"input_rows": len(frame), "usable_rows": len(output), "warmup_or_nonfinite_rows": int((~finite).sum())}

