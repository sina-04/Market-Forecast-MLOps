from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from marketforecast.data import (DataAccessError, TSETMCClient, adjust_prices, clean, collect,
                                 normalize_history, resolve_instruments, validate_snapshot)


def test_official_close_not_last_trade():
    payload = {"closingPriceDaily": [{"dEven": 20250322, "priceFirst": 100, "priceMax": 120,
        "priceMin": 90, "pClosing": 105, "pDrCotVal": 115, "priceYesterday": 99,
        "qTotTran5J": 200, "qTotCap": 21000, "zTotTran": 4}]}
    result = normalize_history(payload, "123")
    assert result.loc[0, "official_close"] == 105
    assert result.loc[0, "last_trade"] == 115
    assert result.loc[0, "volume"] == 200
    assert result.loc[0, "value"] == 21000


def test_exact_company_identity_and_rights_exclusion():
    rows = [
        {"lVal18AFC": "فولاد", "lVal30": "فولاد مبارکه اصفهان", "insCode": "1", "lastDate": 20260101},
        {"lVal18AFC": "فولادح", "lVal30": "فولاد مبارکه اصفهان", "insCode": "2"},
        {"lVal18AFC": "فولاد", "lVal30": "شرکت دیگری", "insCode": "3"}]
    assert [r["insCode"] for r in resolve_instruments({"instrumentSearch": rows}, "فولاد", "فولادمبارکه")] == ["1"]
    with pytest.raises(DataAccessError):
        resolve_instruments({"instrumentSearch": rows}, "فملی", "ملی")


def test_cleaning_dates_duplicates_and_gaps(raw_frame):
    raw_frame.loc[1, "date"] = "2022-01-04"
    raw_frame["volume"] = raw_frame["volume"].astype(object)
    raw_frame.loc[0, "volume"] = "10,000"
    raw = pd.concat([raw_frame.iloc[::-1], raw_frame.iloc[[0]]], ignore_index=True)
    cleaned, rejected, audit = clean(raw)
    assert cleaned["date"].is_monotonic_increasing
    assert cleaned["date"].is_unique
    assert len(cleaned) == len(raw_frame)
    assert audit["reason_counts"] == {"identical_duplicate": 1}
    assert (cleaned["elapsed_days"] > 1).any()
    assert cleaned.loc[0, "jalali_date"] == "1400-10-13"


def test_conflicting_date_quarantined(raw_frame):
    duplicate = raw_frame.iloc[[0]].copy()
    duplicate["volume"] += 1
    cleaned, rejected, audit = clean(pd.concat([raw_frame, duplicate], ignore_index=True))
    assert len(rejected) == 2
    assert not audit["adjustment_safe"]
    assert cleaned["date"].is_unique


@pytest.mark.parametrize("column,value,reason", [
    ("date", "bad", "invalid_date"), ("open", 0, "nonpositive_price"),
    ("low", 99999, "invalid_candle"), ("volume", -1, "negative_activity"),
    ("volume", 0, "trades_without_volume"), ("trade_count", 0, "no_actual_trades"),
    ("official_close", np.nan, "missing_or_nonfinite_numeric"),
    ("previous_official_close", 0, "invalid_adjustment_reference")])
def test_invalid_records_are_audited(raw_frame, column, value, reason):
    raw_frame.loc[3, column] = value
    _, rejected, _ = clean(raw_frame)
    assert reason in rejected.iloc[0]["reason"]


def test_official_close_may_be_outside_traded_range(raw_frame):
    raw_frame.loc[0, "official_close"] = 1100
    cleaned, rejected, _ = clean(raw_frame)
    assert rejected.empty
    assert len(cleaned) == len(raw_frame)


def test_adjustment_handles_split_without_changing_volume(raw_frame):
    raw_frame = raw_frame.iloc[:3].copy()
    raw_frame["official_close"] = [1000, 500, 510]
    raw_frame["previous_official_close"] = [1000, 500, 500]
    cleaned, _, _ = clean(raw_frame)
    result = adjust_prices(cleaned)
    np.testing.assert_allclose(result["adj_official_close"], [500, 500, 510])
    np.testing.assert_allclose(result["adjustment_factor"], [0.5, 1, 1])
    np.testing.assert_array_equal(result["volume"], cleaned["volume"])
    assert result.loc[1, "raw_jump_flag"]


def test_api_timeout_bounded_and_schema_rejected():
    session = Mock()
    session.get.side_effect = requests.Timeout("offline")
    client = TSETMCClient(attempts=3, session=session, sleeper=lambda _: None)
    with pytest.raises(DataAccessError, match="3 attempts"):
        client.get("Instrument/test", "instrumentSearch")
    assert session.get.call_count == 3
    session.get.side_effect = None
    session.get.return_value.json.return_value = {"unexpected": []}
    with pytest.raises(DataAccessError, match="Unexpected TSETMC schema"):
        client.get("Instrument/test", "instrumentSearch")


def test_collection_freezes_completed_session_and_detects_tampering(tmp_path, raw_frame, config):
    # Spread mock source across five years and include today to verify exclusion.
    raw_frame["date"] = pd.date_range("2021-09-26", "2026-09-27", periods=len(raw_frame)).strftime("%Y%m%d")
    inverse = {v: k for k, v in __import__("marketforecast.data", fromlist=["FIELDS"]).FIELDS.items()}
    records = raw_frame.drop(columns="instrument_id").rename(columns=inverse).to_dict("records")
    client = Mock()
    client.get.side_effect = [
        {"instrumentSearch": [{"lVal18AFC": "فولاد", "lVal30": "فولاد مبارکه", "insCode": "123", "lastDate": 20260927}]},
        {"closingPriceDaily": records}]
    path, manifest = collect(tmp_path, config, client, datetime(2026, 9, 27, 10, tzinfo=timezone.utc))
    assert manifest["end_date"] < "2026-09-27"
    assert len(manifest["checksums"]) == 3
    client.get.reset_mock()
    assert collect(tmp_path, config, client)[1] == manifest
    client.get.assert_not_called()
    (path / "raw.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(DataAccessError, match="checksum mismatch"):
        validate_snapshot(path)


def test_collection_failure_not_frozen(tmp_path, config):
    client = Mock()
    client.get.side_effect = DataAccessError("offline")
    with pytest.raises(DataAccessError):
        collect(tmp_path, config, client)
    assert (tmp_path / "data/snapshot/collection_failure.json").exists()
    assert not (tmp_path / "data/snapshot/manifest.json").exists()

