"""Immutable, checksummed release assembly and real-model parity verification."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from marketforecast.io import checksum, read_json, write_json

MODEL_NAME = "foolad-hourly-lstm-replay"


def content_id(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_files(root, files):
    root = Path(root).resolve()
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or checksum(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")


def validate_bundle(root, check_code=True):
    root = Path(root)
    manifest = read_json(root / "release.json")
    if manifest.get("schema_version") != 1 or content_id(manifest["checksums"]) != manifest["release_id"]:
        raise ValueError("Invalid release manifest or fingerprint")
    verify_files(root, manifest["checksums"])
    required = {"model.keras", "scalers.joblib", "model_metadata.json", "replay_index.csv", "reference.json",
                "snapshot/manifest.json", "runtime.json", "feature_order.json", "config.json"}
    if not required.issubset(manifest["checksums"]):
        raise ValueError("Release is missing required inference artifacts")
    if check_code:
        package = Path(__file__).resolve().parents[1]
        for relative, expected in manifest["checksums"].items():
            if relative.startswith("code/"):
                local = package / relative.removeprefix("code/")
                if not local.is_file() or checksum(local) != expected:
                    raise ValueError(f"Serving code differs from approved release: {relative}")
        runtime = read_json(root / "runtime.json")
        for name, version in runtime["packages"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Runtime dependency differs from release: {name}")
        if platform.python_version() != runtime["python"]:
            raise ValueError("Python version differs from approved release")
    return manifest


def build_bundle(source, output):
    source, output = Path(source), Path(output)
    verify_files(source, read_json(source / "bundle_checksums.json"))
    if output.exists():
        raise ValueError("Bundle destination already exists; use a fresh output directory")
    run = source / "training-results/hourly-forecast"
    output.mkdir(parents=True)
    shutil.copytree(source / "data/snapshot", output / "snapshot")
    for name in ["model.keras", "scalers.joblib", "model_metadata.json", "feature_order.json", "config.json"]:
        shutil.copy2(run / name, output / name)
    for name in ["test_predictions.csv", "test_metrics.json", "sequences_test.npz", "completion.json",
                 "source_commit.json", "source_hashes.json", "local_verification.json"]:
        (output / "evidence").mkdir(exist_ok=True)
        shutil.copy2(run / name, output / "evidence" / name)
    package = Path(__file__).resolve().parents[1]
    for path in sorted(package.rglob("*.py")):
        target = output / "code" / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    build_root = package.parents[1]
    (output / "build").mkdir()
    for name in ["Dockerfile", "requirements-service.lock", "pyproject.toml"]:
        shutil.copy2(build_root / name, output / "build" / name)
    runtime = {"python": platform.python_version(), "packages": {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        if d.metadata["Name"].lower() != "marketforecast"
    }}
    write_json(output / "runtime.json", runtime)
    from .runtime import prepare_features
    from marketforecast.sequences import make_plan
    frame, names = prepare_features(output)
    plan = make_plan(frame, read_json(output / "config.json"))
    with np.load(run / "sequences_test.npz", allow_pickle=False) as test:
        positions = test["target_positions"]
        np.testing.assert_array_equal(positions, plan.target_positions[plan.validation_end:])
        index = pd.DataFrame({"as_of": frame.date.iloc[positions - 1].to_numpy(),
                              "target_timestamp": frame.date.iloc[positions].to_numpy(),
                              "position": positions})
    index.to_csv(output / "replay_index.csv", index=False)
    from .monitoring import make_reference
    reference = make_reference(frame.iloc[plan.target_positions[:plan.train_end] - 1][names])
    write_json(output / "reference.json", reference)
    checksums = {str(p.relative_to(output)).replace("\\", "/"): checksum(p)
                 for p in sorted(output.rglob("*")) if p.is_file()}
    release = {"schema_version": 1, "model_name": MODEL_NAME, "release_id": content_id(checksums),
               "checksums": checksums, "supported_windows": len(index)}
    write_json(output / "release.json", release)
    return release


def verify_bundle(root):
    from .runtime import Predictor
    predictor = Predictor(root)
    root = Path(root)
    with np.load(root / "evidence/sequences_test.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(predictor.windows, saved["X"], rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(predictor.previous, saved["persistence"], rtol=1e-10)
    predictions = predictor.predict_batch()
    expected = pd.read_csv(root / "evidence/test_predictions.csv")
    np.testing.assert_array_equal(predictor.index.target_timestamp.to_numpy(), expected.date.to_numpy())
    np.testing.assert_allclose(predictions, expected.lstm, rtol=1e-6, atol=0.001)
    return {"verified": True, "release_id": predictor.release["release_id"], "predictions": len(predictions),
            "max_absolute_difference": float(np.max(np.abs(predictions - expected.lstm))),
            "tolerance": {"rtol": 1e-6, "atol_adjusted_rial": 0.001}}
