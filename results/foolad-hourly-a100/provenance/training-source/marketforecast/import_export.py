"""Freeze user-supplied pytse exports without modifying their original bytes."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import PRICE_COLUMNS, adjust_prices, parse_dates, validate_snapshot
from .io import checksum, write_json

MAPPING = {"adjClose": "official_close", "close": "last_trade",
           "yesterday": "previous_official_close", "count": "trade_count"}


def import_exports(root, config, unadjusted, adjusted, export_script=None, now=None):
    snapshot = Path(root) / "data" / "snapshot"
    if (snapshot / "manifest.json").exists():
        raise ValueError("Snapshot already frozen; use a new root to import another snapshot")
    raw = pd.read_csv(unadjusted).rename(columns=MAPPING)
    comparison = pd.read_csv(adjusted).rename(columns=MAPPING)
    required = ["date", *PRICE_COLUMNS, "previous_official_close", "volume", "value", "trade_count"]
    if set(required) - set(raw) or set(required) - set(comparison):
        raise ValueError("Export is missing required pytse fields")
    raw["date"] = parse_dates(raw.date)
    comparison["date"] = parse_dates(comparison.date)
    for frame in [raw, comparison]:
        if frame.date.isna().any() or frame.date.duplicated().any():
            raise ValueError("Invalid or duplicate export dates require source review")
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    np.testing.assert_array_equal(raw.date, comparison.date)
    for column in ["volume", "value", "trade_count"]:
        np.testing.assert_array_equal(raw[column], comparison[column])
    if "jdate" in raw:
        import jdatetime
        expected = raw.date.map(lambda d: str(jdatetime.date.fromgregorian(date=d.date())))
        np.testing.assert_array_equal(expected, raw.jdate)
    # Keep every valid official-close reference in the adjustment chain, even when
    # another candle field is invalid. Cleaning will quarantine the entire candle.
    basis = adjust_prices(raw)
    discrepancies = {}
    for column in PRICE_COLUMNS:
        difference = (basis[f"adj_{column}"] - comparison[column]).abs()
        discrepancies[column] = float(difference.max())
        if difference.max() > 0.500001:
            raise ValueError(f"Supplied adjusted {column} disagrees with backward adjustment")
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now.astimezone(ZoneInfo("Asia/Tehran")).date())
    eligible = raw.loc[(raw.date < today) & (raw.trade_count > 0), "date"]
    end = eligible.max()
    start = end - pd.DateOffset(years=config["years"])
    if eligible.empty or eligible.min() > start + pd.Timedelta(days=30):
        raise ValueError("Insufficient completed-session history")
    window = raw.loc[raw.date.between(start, end), required].copy()
    window["instrument_id"] = "46348559193224090"
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(unadjusted, snapshot / "source_unadjusted.csv")
    shutil.copy2(adjusted, snapshot / "source_adjusted.csv")
    if export_script:
        shutil.copy2(export_script, snapshot / "source_export_script.py")
    window.to_csv(snapshot / "raw.csv", index=False)
    # Recompute on the frozen window: its final basis is the last completed session.
    frozen_basis = adjust_prices(window)
    frozen_basis[["date", "adjustment_factor"]].to_csv(snapshot / "adjustment_basis.csv", index=False)
    write_json(snapshot / "export_validation.json", {
        "full_history_rows": len(raw), "frozen_rows": len(window),
        "adjusted_export_max_absolute_discrepancy_rial": discrepancies,
        "rounding_tolerance_rial": 0.500001,
        "activity_columns_identical": True, "jalali_dates_checked": "jdate" in raw,
        "identity_basis": "User export script explicitly downloads ticker فولاد; pytse bundled identifier",
        "http_responses_available": False,
    })
    manifest = {
        "symbol": config["symbol"], "company_contains": config["company_contains"],
        "years": config["years"], "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
        "instrument_ids": ["46348559193224090"], "isin": "IRO1FOLD0001",
        "company": "فولاد مبارکه اصفهان", "retrieved_at_utc": "Not recorded by source exporter",
        "imported_at_utc": now.isoformat(), "source": "TSETMC via user-supplied pytse-client CSV exports",
        "adapter_version": "pytse-export-import/0.1.0", "source_version": "Not recorded by source exporter",
        "mapping_reference": "https://github.com/Glyphack/pytse-client",
        "field_mapping": MAPPING, "completion_policy": "Strictly before current Tehran date",
        "checksums": {p.name: checksum(p) for p in snapshot.iterdir() if p.is_file()},
    }
    write_json(snapshot / "manifest.json", manifest)
    return snapshot, validate_snapshot(snapshot)
