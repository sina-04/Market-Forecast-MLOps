"""Opt-in real TensorFlow execution on toy data, never market performance evidence."""
import os
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from marketforecast.data import FIELDS, collect
from marketforecast.io import read_json
from marketforecast.workflow import export_bundle, run, verify_run


@pytest.mark.skipif(os.getenv("MARKETFORECAST_TRAIN_TEST") != "1", reason="Set MARKETFORECAST_TRAIN_TEST=1 for TensorFlow integration")
def test_saved_models_scalers_reports_archive_and_resume(tmp_path, raw_frame, config, monkeypatch):
    pytest.importorskip("tensorflow")
    raw_frame["date"] = pd.date_range("2021-01-01", "2026-09-26", periods=len(raw_frame)).strftime("%Y%m%d")
    records = raw_frame.drop(columns="instrument_id").rename(columns={v: k for k, v in FIELDS.items()}).to_dict("records")
    client = Mock()
    client.get.side_effect = [
        {"instrumentSearch": [{"lVal18AFC": "فولاد", "lVal30": "فولاد مبارکه", "insCode": "123", "lastDate": 20260926}]},
        {"closingPriceDaily": records}]
    config = {**config, "epochs": 2, "patience": 1}
    collect(tmp_path, config, client, datetime(2026, 9, 27, tzinfo=timezone.utc))
    output = run(tmp_path, config, acquire=False)
    assert read_json(output / "completion.json")["saved_model_reload_verified"]
    assert verify_run(output)["verified"]
    predicted = pd.read_csv(output / "test_predictions.csv")
    assert np.isfinite(predicted.drop(columns="date")).all().all()
    assert len(read_json(output / "validation_search.json")) == 6
    for filename in ["model.keras", "model.h5", "report.md", "test_forecasts.png", "residuals.png", "loss_selected_lstm.png"]:
        assert (output / filename).stat().st_size > 0
    from zipfile import ZipFile
    with ZipFile(export_bundle(tmp_path, output)) as archive:
        assert "data/snapshot/raw.csv" in archive.namelist()
        assert f"runs/{output.name}/model.h5" in archive.namelist()
    # Completed-run reuse must not retrain or reselect models.
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed run attempted to train again")
    monkeypatch.setattr("marketforecast.training.train", forbidden)
    assert run(tmp_path, config, acquire=False) == output
