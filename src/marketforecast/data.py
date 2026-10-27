"""Bounded, HTTPS-only TSETMC acquisition and auditable normalization.

Mapping reference: ARahimiQuant/finpy-tse, get_price_history.
FinPy Final = pClosing (official closing); Close = pDrCotVal (last trade).
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

import jdatetime
import numpy as np
import pandas as pd
import requests

from .io import checksum, read_json, write_json

BASE_URL = "https://cdn.tsetmc.com/api"
FIELDS = {
    "dEven": "date", "priceFirst": "open", "priceMax": "high",
    "priceMin": "low", "pClosing": "official_close", "pDrCotVal": "last_trade",
    "priceYesterday": "previous_official_close", "qTotTran5J": "volume",
    "qTotCap": "value", "zTotTran": "trade_count",
}
NUMERIC = list(FIELDS.values())[1:]
PRICE_COLUMNS = ["open", "high", "low", "official_close", "last_trade"]


class DataAccessError(RuntimeError):
    """Live collection failed; cached validated data may still be used."""


def normalize_name(value):
    return re.sub(r"[\s\u200c\u200f]", "", str(value).replace("ي", "ی").replace("ك", "ک"))


class TSETMCClient:
    def __init__(self, timeout=15, attempts=3, session=None, sleeper=time.sleep):
        self.timeout = timeout
        self.attempts = attempts
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "MarketForecast/0.1 (historical research)", "Accept": "application/json"})
        self.sleeper = sleeper

    def get(self, path, key):
        errors = []
        for attempt in range(self.attempts):
            try:
                response = self.session.get(f"{BASE_URL}/{path}", timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
                    raise ValueError(f"Unexpected TSETMC schema: missing list '{key}'")
                return payload
            except (requests.RequestException, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}")
                if attempt + 1 < self.attempts:
                    self.sleeper(min(2 ** attempt, 4))
        raise DataAccessError(f"TSETMC {path} failed after {self.attempts} attempts. " + " | ".join(errors))


def resolve_instruments(payload, symbol, company_contains):
    matches = [r for r in payload["instrumentSearch"]
               if normalize_name(r.get("lVal18AFC", "")) == normalize_name(symbol)
               and normalize_name(company_contains) in normalize_name(r.get("lVal30", ""))]
    if not matches:
        raise DataAccessError(f"No exact equity ticker/company match for {symbol}; no fuzzy fallback used.")
    ids = {}
    for record in matches:
        code = str(record["insCode"])
        if not code.isdigit():
            raise DataAccessError("Invalid instrument identifier")
        ids[code] = record
    return sorted(ids.values(), key=lambda r: int(r.get("lastDate") or 0), reverse=True)


def normalize_history(payload, instrument_id):
    records = payload["closingPriceDaily"]
    if not records:
        raise DataAccessError(f"Empty history for instrument {instrument_id}")
    frame = pd.DataFrame(records)
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise DataAccessError(f"Missing required history fields: {sorted(missing)}")
    frame = frame[list(FIELDS)].rename(columns=FIELDS)
    # Preserve malformed values for the audit instead of silently dropping them here.
    frame["instrument_id"] = str(instrument_id)
    return frame


def parse_dates(values):
    text = values.astype(str).str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    result = pd.to_datetime(text.where(~compact), format="%Y-%m-%d", errors="coerce")
    result.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return result


def validate_snapshot(snapshot):
    snapshot = Path(snapshot)
    manifest = read_json(snapshot / "manifest.json")
    for name, expected in manifest["checksums"].items():
        if checksum(snapshot / name) != expected:
            raise DataAccessError(f"Frozen snapshot checksum mismatch: {name}")
    return manifest


def collect(root, config, client=None, now=None):
    snapshot = Path(root) / "data" / "snapshot"
    if (snapshot / "manifest.json").exists():
        return snapshot, validate_snapshot(snapshot)
    if config.get("frequency") == "hourly":
        raise ValueError("Hourly data must be imported with import-hourly; daily collection cannot substitute")
    snapshot.mkdir(parents=True, exist_ok=True)
    client = client or TSETMCClient(config["request_timeout"], config["request_attempts"])
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now.astimezone(ZoneInfo("Asia/Tehran")).date())
    try:
        search = client.get(f"Instrument/GetInstrumentSearch/{quote(config['symbol'])}", "instrumentSearch")
        write_json(snapshot / "instrument_search.json", search)
        instruments = resolve_instruments(search, config["symbol"], config["company_contains"])
        frames, sources = [], ["instrument_search.json"]
        for record in instruments:
            code = str(record["insCode"])
            response = client.get(f"ClosingPrice/GetClosingPriceDailyList/{code}/0", "closingPriceDaily")
            filename = f"history_{code}.json"
            write_json(snapshot / filename, response)
            sources.append(filename)
            frames.append(normalize_history(response, code))
        history = pd.concat(frames, ignore_index=True)
        dates = parse_dates(history["date"])
        traded = pd.to_numeric(history["trade_count"], errors="coerce") > 0
        eligible = dates[(dates < today) & traded]
        if eligible.empty:
            raise DataAccessError("No completed traded session before today's Tehran date")
        end = eligible.max()
        start = end - pd.DateOffset(years=config["years"])
        if eligible.min() > start + pd.Timedelta(days=30):
            raise DataAccessError("History does not cover the requested five-year interval")
        # Source JSON retains all fetched rows. CSV preserves malformed dates for cleaning audit.
        history = history.loc[dates.between(start, end) | dates.isna()].copy()
        history.to_csv(snapshot / "raw.csv", index=False, encoding="utf-8")
        sources.append("raw.csv")
        manifest = {
            "symbol": config["symbol"], "company_contains": config["company_contains"],
            "retrieved_at_utc": now.isoformat(), "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(), "years": config["years"],
            "instrument_ids": [str(r["insCode"]) for r in instruments],
            "source": BASE_URL, "adapter_version": "0.1.0",
            "mapping_reference": "https://github.com/ARahimiQuant/finpy-tse",
            "completion_policy": "Strictly before today's Tehran date; no intraday candles",
            "checksums": {name: checksum(snapshot / name) for name in sources},
        }
        write_json(snapshot / "manifest.json", manifest)
        return snapshot, manifest
    except Exception as error:
        write_json(snapshot / "collection_failure.json", {"at_utc": now.isoformat(), "error": str(error)})
        raise


def clean(raw):
    frame = raw.copy()
    frame["source_row"] = np.arange(len(frame))
    frame["date"] = parse_dates(frame["date"])
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    reasons = pd.Series("", index=frame.index)

    def flag(mask, reason):
        nonlocal reasons
        reasons.loc[mask] = reasons.loc[mask].map(lambda old: f"{old};{reason}".strip(";"))

    flag(frame["date"].isna(), "invalid_date")
    flag(~np.isfinite(frame[NUMERIC]).all(axis=1), "missing_or_nonfinite_numeric")
    flag((frame[PRICE_COLUMNS] <= 0).any(axis=1), "nonpositive_price")
    flag(frame["previous_official_close"] <= 0, "invalid_adjustment_reference")
    flag((frame[["volume", "value", "trade_count"]] < 0).any(axis=1), "negative_activity")
    flag(frame["trade_count"] == 0, "no_actual_trades")
    flag((frame["trade_count"] > 0) & (frame["volume"] == 0), "trades_without_volume")
    flag((frame[["volume", "trade_count"]] % 1 != 0).any(axis=1), "fractional_activity")
    candle_invalid = (frame["high"] < frame["low"])
    for column in ["open", "last_trade"]:
        candle_invalid |= ~frame[column].between(frame["low"], frame["high"])
    # Official close is weighted/subject to base-volume rules; it need not lie inside traded range.
    flag(candle_invalid, "invalid_candle")
    valid = frame.loc[reasons == ""].sort_values(["date", "source_row"])
    compare = ["date"] + NUMERIC
    duplicates = valid.duplicated(compare, keep="first")
    flag(frame.index.isin(valid.index[duplicates]), "identical_duplicate")
    remaining = valid.loc[~duplicates]
    conflicts = remaining["date"].duplicated(keep=False)
    flag(frame.index.isin(remaining.index[conflicts]), "conflicting_duplicate_date")
    rejected = frame.loc[reasons != ""].copy()
    rejected["reason"] = reasons.loc[rejected.index]
    cleaned = frame.loc[reasons == ""].sort_values("date").reset_index(drop=True)
    if len(cleaned) < 2:
        raise ValueError("Fewer than two valid sessions after cleaning")
    # Removing a traded row prevents an honest adjustment chain and shifts forecast horizons.
    critical = rejected["reason"].str.contains("conflicting_duplicate_date|invalid_adjustment_reference")
    adjustment_safe = not critical.any()
    cleaned["jalali_date"] = cleaned["date"].map(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    cleaned["elapsed_days"] = cleaned["date"].diff().dt.days.fillna(0).astype(int)
    audit = {
        "raw_rows": len(raw), "clean_rows": len(cleaned), "quarantined_rows": len(rejected),
        "reason_counts": rejected["reason"].str.split(";").explode().value_counts().to_dict(),
        "adjustment_safe": adjustment_safe,
    }
    return cleaned, rejected, audit


def adjust_prices(cleaned):
    """Backward adjustment using next session's previous-close/current official-close.

    Uses later information. This fixed snapshot is explicitly retrospective.
    Float precision is retained; original volume is never price-adjusted.
    """
    result = cleaned.copy()
    ratios = result["previous_official_close"].shift(-1) / result["official_close"]
    ratios.iloc[-1] = 1.0
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError("Invalid corporate-action adjustment coefficient")
    result["adjustment_factor"] = ratios.iloc[::-1].cumprod().iloc[::-1]
    for column in PRICE_COLUMNS:
        result[f"adj_{column}"] = result[column] * result["adjustment_factor"]
    result["raw_jump_flag"] = result["official_close"].pct_change().abs() > 0.15
    result["adjustment_event"] = ~np.isclose(ratios, 1, atol=1e-8)
    return result
