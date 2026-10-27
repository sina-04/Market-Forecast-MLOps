"""Freeze hourly trade bars and apply a daily corporate-action basis once per day."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import jdatetime
import numpy as np
import pandas as pd

from .data import adjust_prices, validate_snapshot
from .import_export import MAPPING
from .io import checksum, write_json

BAR_COLUMNS = ["open", "high", "low", "hourly_last_price", "volume", "returned_trade_count"]


def import_hourly(root, config, hourly, daily, coverage, export_script=None, trades=None, now=None):
    if config.get("frequency") != "hourly":
        raise ValueError("Hourly import requires frequency=hourly")
    snapshot = Path(root) / "data" / "snapshot"
    if (snapshot / "manifest.json").exists():
        raise ValueError("Snapshot already frozen; use a new root")
    bars = pd.read_csv(hourly)
    if {"datetime", *BAR_COLUMNS} - set(bars):
        raise ValueError("Hourly export is missing required fields")
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    if bars.datetime.isna().any() or bars.datetime.duplicated().any():
        raise ValueError("Invalid or duplicate hourly timestamps")
    if bars.datetime.dt.tz is not None or (bars.datetime != bars.datetime.dt.floor("h")).any():
        raise ValueError("Expected whole-hour naive exchange-local timestamps")
    bars = bars.sort_values("datetime").reset_index(drop=True)
    now = now or datetime.now(timezone.utc)
    today = pd.Timestamp(now.astimezone(ZoneInfo("Asia/Tehran")).date())
    if (bars.datetime.dt.normalize() >= today).any():
        raise ValueError("Hourly source contains an incomplete/current or future session")
    log = pd.read_csv(coverage)
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    if log.date.isna().any() or log.date.duplicated().any() or not log.status.isin(["downloaded", "unresolved"]).all():
        raise ValueError("Invalid coverage log")
    counts = bars.groupby(bars.datetime.dt.normalize()).size()
    success = log.loc[log.status.eq("downloaded")].set_index("date").sort_index()
    np.testing.assert_array_equal(counts.index, success.index)
    np.testing.assert_array_equal(counts.values, pd.to_numeric(success.hourly_bars).values)
    history = pd.read_csv(daily).rename(columns=MAPPING)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    if history.date.isna().any() or history.date.duplicated().any():
        raise ValueError("Invalid daily adjustment dates")
    history = history.loc[history.date <= bars.datetime.max().normalize()].sort_values("date").reset_index(drop=True)
    # Only daily official-close references determine actions; hourly last prices never do.
    basis = adjust_prices(history)[["date", "adjustment_factor"]]
    if not counts.index.isin(basis.date).all():
        raise ValueError("Hourly bars lack daily adjustment references")
    eligible_daily = history.loc[history.date.between(log.date.min(), log.date.max()) & history.trade_count.gt(0), "date"]
    if not eligible_daily.isin(log.date).all():
        raise ValueError("Coverage log omits traded daily sessions")
    inventory = {}
    if trades is not None:
        for day, row in success.iterrows():
            path = Path(trades) / f"{day.date()}.csv"
            ticks = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index(kind="stable")
            if ticks.empty or ticks.index.isna().any() or not (ticks.index.normalize() == day).all():
                raise ValueError(f"Invalid raw trades for {day.date()}")
            grouped = ticks.resample("1h", closed="left", label="left")
            reconstructed = grouped.agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
                                        hourly_last_price=("price", "last"), volume=("volume", "sum"),
                                        returned_trade_count=("price", "size"))
            reconstructed = reconstructed.loc[reconstructed.returned_trade_count > 0]
            exported = bars.loc[bars.datetime.dt.normalize().eq(day)].set_index("datetime")
            np.testing.assert_array_equal(reconstructed.index, exported.index)
            np.testing.assert_allclose(reconstructed[BAR_COLUMNS], exported[BAR_COLUMNS], rtol=1e-12, atol=1e-8)
            if len(ticks) != int(row.returned_trades):
                raise ValueError("Raw trade count disagrees with coverage")
            inventory[path.name] = checksum(path)
    snapshot.mkdir(parents=True, exist_ok=True)
    for source, name in [(hourly, "source_hourly.csv"), (daily, "source_daily.csv"), (coverage, "coverage.csv")]:
        shutil.copy2(source, snapshot / name)
    if export_script:
        shutil.copy2(export_script, snapshot / "source_export_script.py")
    bars.rename(columns={"datetime": "date"}).to_csv(snapshot / "raw.csv", index=False)
    basis.to_csv(snapshot / "adjustment_basis.csv", index=False)
    write_json(snapshot / "raw_trade_checksums.json", inventory)
    write_json(snapshot / "export_validation.json", {
        "raw_trade_aggregation_verified": trades is not None, "raw_trade_files": len(inventory),
        "raw_trade_cache_in_bundle": False, "hourly_rows": len(bars),
        "downloaded_dates": len(success), "unresolved_dates": int(log.status.eq("unresolved").sum()),
        "unresolved_errors": log.loc[log.status.eq("unresolved"), ["date", "error"]].astype(str).to_dict("records"),
    })
    manifest = {
        "symbol": config["symbol"], "company_contains": config["company_contains"], "years": config["years"],
        "frequency": "hourly", "timezone": "Asia/Tehran", "timezone_basis": "Naive exchange-local timestamps from exporter; no timezone conversion",
        "target": "next observed traded hourly bar adjusted last price", "price_column": "adj_hourly_last_price",
        "start_date": str(bars.datetime.min()), "end_date": str(bars.datetime.max()),
        "instrument_ids": ["46348559193224090"], "source": "User pytse-client hourly bars aggregated from returned trades",
        "retrieved_at_utc": "Not recorded by exporter", "imported_at_utc": now.isoformat(),
        "completion_policy": "Completed dates only; next observed bar may follow an overnight/weekend gap",
        "adjustment_policy": "Daily next-session previous official close / official close, applied uniformly within each day; retrospective",
        "checksums": {p.name: checksum(p) for p in snapshot.iterdir() if p.is_file()},
    }
    write_json(snapshot / "manifest.json", manifest)
    return snapshot, validate_snapshot(snapshot)


def clean_hourly(raw, snapshot):
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for name in BAR_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    prices = ["open", "high", "low", "hourly_last_price"]
    valid = frame.date.notna() & np.isfinite(frame[BAR_COLUMNS]).all(axis=1)
    valid &= (frame[prices] > 0).all(axis=1) & (frame[["volume", "returned_trade_count"]] > 0).all(axis=1)
    valid &= (frame[["volume", "returned_trade_count"]] % 1 == 0).all(axis=1)
    valid &= frame.high.ge(frame.low)
    for name in ["open", "hourly_last_price"]:
        valid &= frame[name].between(frame.low, frame.high)
    valid &= ~frame.date.duplicated(keep=False)
    if not valid.all():
        raise ValueError(f"Hourly data audit failed for {int((~valid).sum())} rows; repair source before training")
    frame = frame.sort_values("date").reset_index(drop=True)
    snapshot = Path(snapshot)
    basis = pd.read_csv(snapshot / "adjustment_basis.csv", parse_dates=["date"])
    # Recalculate frozen daily basis to detect an inconsistent supplied adjustment chain.
    daily = pd.read_csv(snapshot / "source_daily.csv").rename(columns=MAPPING)
    daily["date"] = pd.to_datetime(daily.date)
    daily = daily.loc[daily.date <= frame.date.max().normalize()].sort_values("date").reset_index(drop=True)
    expected = adjust_prices(daily)
    np.testing.assert_array_equal(basis.date, expected.date)
    np.testing.assert_allclose(basis.adjustment_factor, expected.adjustment_factor, rtol=1e-12)
    frame["session_date"] = frame.date.dt.normalize()
    frame = frame.merge(basis.rename(columns={"date": "session_date"}), on="session_date", validate="many_to_one")
    if not np.isfinite(frame.adjustment_factor).all():
        raise ValueError("Missing hourly corporate-action basis")
    for name in prices:
        frame[f"adj_{name}"] = frame[name] * frame.adjustment_factor
    coverage = pd.read_csv(snapshot / "coverage.csv", parse_dates=["date"])
    unresolved = coverage.loc[coverage.status.eq("unresolved"), "date"].to_numpy(dtype="datetime64[ns]")
    # A change in segment creates a source-number gap; sequences cannot cross known missing days.
    frame["segment"] = np.searchsorted(np.sort(unresolved), frame.session_date.to_numpy(), side="right")
    frame["session_number"] = np.arange(len(frame)) + frame.segment
    frame["elapsed_hours"] = frame.date.diff().dt.total_seconds().div(3600).fillna(0)
    frame["elapsed_days"] = frame.elapsed_hours / 24
    frame["jalali_date"] = frame.date.map(lambda d: str(jdatetime.date.fromgregorian(date=d.date())))
    frame["raw_jump_flag"] = frame.hourly_last_price.pct_change().abs() > .15
    frame["adjustment_event"] = frame.adjustment_factor.diff().abs() > 1e-8
    audit = {"raw_rows": len(raw), "clean_rows": len(frame), "quarantined_rows": 0, "reason_counts": {},
             "adjustment_safe": True, "adjustment_basis": "Verified complete daily official-close chain applied once per session",
             "unresolved_dates": len(unresolved), "segments": int(frame.segment.nunique()),
             "observed_hours": sorted(frame.date.dt.hour.unique().tolist()),
             "late_hour_bars": int(frame.date.dt.hour.ge(13).sum()),
             "late_hour_policy": "Preserved as returned; no unverified session-hour filtering"}
    return frame, frame.iloc[:0].assign(reason=pd.Series(dtype=str)), audit
