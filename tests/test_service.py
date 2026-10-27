"""Operational checks; numerical parity uses the real, frozen market artifacts."""
from pathlib import Path
import sqlite3
import uuid

import pytest
pytest.importorskip("fastapi", reason="Service dependencies are tested in the production container")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from marketforecast.io import read_json, write_json
from marketforecast.service.api import create_app
from marketforecast.service.audit import AuditStore
from marketforecast.service.bundle import build_bundle, validate_bundle, verify_bundle
from marketforecast.service.monitoring import Monitor, make_reference, psi
from marketforecast.service.runtime import Predictor, local_timestamp, prepare_features

SOURCE = Path("results/foolad-hourly-a100")
KEY = "test-only-key-not-a-production-secret"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    if not SOURCE.exists():
        pytest.skip("Real hourly publication is not available")
    pytest.importorskip("tensorflow")
    path = tmp_path_factory.mktemp("service") / "bundle"
    build_bundle(SOURCE, path)
    return path


@pytest.fixture(scope="module")
def predictor(bundle):
    return Predictor(bundle)


@pytest.fixture
def client(bundle, tmp_path):
    registration = tmp_path / "registration.json"
    write_json(registration, {"release_id": read_json(bundle / "release.json")["release_id"],
                              "verified": True, "registry_version": "test"})
    with TestClient(create_app(bundle, tmp_path / "audit.sqlite", KEY, registration)) as client:
        yield client


def test_all_535_direct_predictions_match_frozen_research(bundle):
    result = verify_bundle(bundle)
    assert result["verified"] and result["predictions"] == 535


def test_bundle_detects_corruption_and_contract_changes(bundle, tmp_path):
    import shutil
    from marketforecast.service.bundle import content_id
    damaged = tmp_path / "damaged"
    shutil.copytree(bundle, damaged)
    (damaged / "model.keras").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(damaged)
    (damaged / "model.keras").unlink()
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(damaged)
    shutil.copy2(bundle / "model.keras", damaged / "model.keras")
    metadata = read_json(damaged / "model_metadata.json")
    metadata["selected"]["length"] = 24
    write_json(damaged / "model_metadata.json", metadata)
    from marketforecast.io import checksum
    release = read_json(damaged / "release.json")
    release["checksums"]["model_metadata.json"] = checksum(damaged / "model_metadata.json")
    release["release_id"] = content_id(release["checksums"])
    write_json(damaged / "release.json", release)
    with pytest.raises(ValueError, match="contract"):
        Predictor(damaged)


def test_all_535_http_predictions_match_frozen_research(client, bundle):
    entries = client.get("/v1/replay/timestamps", headers={"X-API-Key": KEY}).json()["timestamps"]
    saved = pd.read_csv(bundle / "evidence/test_predictions.csv")
    forecasts = []
    for entry in entries:
        response = client.post("/v1/predict", json={"as_of": entry["as_of"]}, headers={"X-API-Key": KEY})
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["mode"] == "historical_replay" and value["unit"] == "retrospectively adjusted rial"
        assert "actual" not in value
        forecasts.append(value["predicted_adjusted_price"])
    np.testing.assert_allclose(forecasts, saved.lstm, rtol=1e-6, atol=.001)
    assert client.app.state.store.count() == 535


def test_http_validation_auth_and_health(client):
    headers = {"X-API-Key": KEY}
    entry = client.get("/v1/replay/timestamps", headers=headers).json()["timestamps"][0]
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200
    assert client.get("/v1/model").status_code == 401
    assert client.get("/v1/model", headers={"X-API-Key": "wrong"}).status_code == 401
    for body in [{}, {"as_of": "2026-01-01"}, {"as_of": 123}, {"as_of": "invalid"},
                 {"as_of": entry["as_of"], "extra": True}]:
        assert client.post("/v1/predict", json=body, headers=headers).status_code == 422
    assert client.post("/v1/predict", json={"as_of": "2001-01-01T09:00:00+03:30"}, headers=headers).status_code == 404
    assert "X-API-Key" in client.get("/openapi.json").text
    assert "forecast_ready" in client.get("/metrics").text


def test_failure_to_audit_never_returns_success(client, monkeypatch):
    entry = client.app.state.predictor.timestamps()[0]
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("Test unavailable storage")
    monkeypatch.setattr(client.app.state.store, "record", fail)
    response = client.post("/v1/predict", json={"as_of": entry["as_of"]}, headers={"X-API-Key": KEY})
    assert response.status_code == 503


def test_readiness_checks_writable_audit_store(client, monkeypatch):
    def fail():
        raise sqlite3.OperationalError("Read-only test storage")
    monkeypatch.setattr(client.app.state.store, "ready", fail)
    assert client.get("/health/ready").status_code == 503
    assert "forecast_ready 0.0" in client.get("/metrics").text


def test_startup_requires_secret_and_matching_registration(bundle, tmp_path):
    with pytest.raises(ValueError, match="24 characters"):
        with TestClient(create_app(bundle, tmp_path / "state.sqlite", "short")):
            pass
    registration = tmp_path / "wrong.json"
    write_json(registration, {"release_id": "wrong", "verified": True, "registry_version": "test"})
    with pytest.raises(ValueError, match="registration differs"):
        with TestClient(create_app(bundle, tmp_path / "state.sqlite", KEY, registration)):
            pass


def test_nonfinite_model_output_is_rejected(client, monkeypatch):
    predictor = client.app.state.predictor
    class Nonfinite:
        def __call__(self, x, training=False):
            import tensorflow as tf
            return tf.constant([[float("nan")]])
    monkeypatch.setattr(predictor, "model", Nonfinite())
    response = client.post("/v1/predict", json={"as_of": predictor.timestamps()[0]["as_of"]}, headers={"X-API-Key": KEY})
    assert response.status_code == 500


def test_timezone_equivalence_and_naive_rejection(predictor):
    entry = predictor.timestamps()[0]["as_of"]
    utc = pd.Timestamp(entry).tz_convert("UTC").isoformat()
    assert predictor.predict(entry) == predictor.predict(utc)
    with pytest.raises(ValueError, match="offset"):
        local_timestamp("2026-01-01 09:00:00")


def test_rolling_diagnostics_deduplication_and_sqlite_backup(predictor, tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite")
    monitor = Monitor(predictor, store)
    assert monitor.evaluate()["status"] == "insufficient_samples"
    for entry in predictor.timestamps()[:101]:
        result = predictor.predict(entry["as_of"])
        for _ in range(2):
            store.record(str(uuid.uuid4()), predictor.release["release_id"], 200, .01, result["input_timestamp"], result)
    report = monitor.evaluate()
    assert report["samples"] == 100 and len(report["psi"]) == 49
    assert report["metrics"]["lstm"]["RMSE"] > 0
    backup = store.backup(tmp_path / "restored.sqlite")
    assert AuditStore(backup).count() == 202
    assert len(AuditStore(backup).latest(predictor.release["release_id"])) == 100


def test_psi_handles_constant_features_and_detects_shift():
    frame = pd.DataFrame({"constant": np.ones(100), "varying": np.linspace(0, 1, 100)})
    reference = make_reference(frame)
    assert psi(frame.constant, reference["constant"]) == pytest.approx(0)
    assert psi(frame.varying, reference["varying"]) == pytest.approx(0)
    assert psi(np.full(100, 10), reference["varying"]) > .2


def test_restore_checksums_and_refuses_existing_data(tmp_path):
    from marketforecast.service.operations import restore
    from marketforecast.io import checksum
    source = tmp_path / "backup"
    (source / "state").mkdir(parents=True)
    AuditStore(source / "state/audit.sqlite")
    (source / "mlflow").mkdir()
    (source / "mlflow/mlflow.db").write_bytes(b"fixture-only")
    (source / "registry").mkdir()
    write_json(source / "registry/registration.json", {"registry_version": "test"})
    write_json(source / "checksums.json", {str(p.relative_to(source)): checksum(p) for p in source.rglob("*") if p.is_file()})
    destinations = [tmp_path / name for name in ["restored-state", "restored-mlflow", "restored-registry"]]
    assert restore(source, *destinations)["restored"]
    with pytest.raises(ValueError, match="empty"):
        restore(source, *destinations)
    (source / "mlflow/mlflow.db").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        restore(source, *[tmp_path / f"other-{i}" for i in range(3)])


def test_features_are_prefix_invariant_and_replay_windows_respect_gaps(bundle, predictor):
    from marketforecast.hourly import clean_hourly
    from marketforecast.features import engineer
    raw = pd.read_csv(bundle / "snapshot/raw.csv")
    cleaned, _, _ = clean_hourly(raw, bundle / "snapshot")
    full, names = prepare_features(bundle)
    # Recursive EMA/Wilder state must start with complete available segment history.
    for position in [int(predictor.index.position.iloc[0]), int(predictor.index.position.iloc[-1])]:
        end_date = full.date.iloc[position - 1]
        prefix, prefix_names, _ = engineer(cleaned.loc[cleaned.date <= pd.Timestamp(end_date)])
        assert prefix_names == names
        np.testing.assert_allclose(prefix[names].to_numpy(), full.loc[full.date <= end_date, names].to_numpy(), atol=1e-10)
    assert full.segment.nunique() > 1
    positions = predictor.index.position.to_numpy(int)
    assert any(np.diff(pd.to_datetime(full.date.iloc[t-60:t+1])).max() > np.timedelta64(1, "h") for t in positions)
