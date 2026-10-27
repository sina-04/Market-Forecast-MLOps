"""MLflow packaging, idempotent registration and explicit model promotion."""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from marketforecast.io import read_json, write_json
from .bundle import MODEL_NAME, verify_bundle
from .runtime import Predictor


class ReplayModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = Predictor(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        return pd.DataFrame([self.predictor.predict(value) for value in model_input["as_of"]])


def register(root, uri, output, promote=False):
    verified = verify_bundle(root)
    release_id = verified["release_id"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name = '{MODEL_NAME}'")
    matching = [v for v in versions if v.tags.get("release_id") == release_id and v.tags.get("verified") == "true"]
    if matching:
        version = matching[0]
    else:
        mlflow.set_experiment("foolad-hourly-production")
        predictor = Predictor(root)
        example = pd.DataFrame({"as_of": [predictor.timestamps()[0]["as_of"]]})
        expected = pd.DataFrame([predictor.predict(example.as_of[0])])
        runtime = read_json(Path(root) / "runtime.json")
        with mlflow.start_run(tags={"release_id": release_id, "mode": "historical_replay"}):
            mlflow.log_params({"sequence_length": 60, "features": 49, "units": 32,
                               "target_mode": "log_return", "snapshot": predictor.metadata["adjustment_basis_date"]})
            scores = read_json(Path(root) / "evidence/test_metrics.json")
            mlflow.log_metrics({f"original_test_{model}_{key}": value
                                for model, values in scores.items() for key, value in values.items()})
            snapshot = Path(root) / "snapshot/raw.csv"
            dataset = mlflow.data.from_pandas(pd.read_csv(snapshot), source=str(snapshot), name="frozen_hourly_snapshot")
            mlflow.log_input(dataset, context="historical_replay")
            info = mlflow.pyfunc.log_model(
                name="predictor", python_model=ReplayModel(), artifacts={"bundle": str(Path(root).resolve())},
                code_paths=[str(Path(__file__).resolve().parents[1])],
                signature=mlflow.models.infer_signature(example, expected), input_example=example,
                pip_requirements=[f"{name}=={value}" for name, value in sorted(runtime["packages"].items())])
            # Exercise the logged wrapper before granting it a verified registry version.
            restored = mlflow.pyfunc.load_model(info.model_uri)
            actual = restored.predict(pd.DataFrame({"as_of": [t["as_of"] for t in predictor.timestamps()]}))
            saved = pd.read_csv(Path(root) / "evidence/test_predictions.csv")
            np.testing.assert_allclose(actual.predicted_adjusted_price, saved.lstm, rtol=1e-6, atol=.001)
            version = mlflow.register_model(info.model_uri, MODEL_NAME)
            client.set_model_version_tag(MODEL_NAME, version.version, "release_id", release_id)
            client.set_model_version_tag(MODEL_NAME, version.version, "verified", "true")
            client.set_model_version_tag(MODEL_NAME, version.version, "mode", "historical_replay")
    client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
    if promote:
        current = client.get_registered_model(MODEL_NAME).aliases.get("champion")
        if current and str(current) != str(version.version):
            client.set_registered_model_alias(MODEL_NAME, "previous", current)
        client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
    record = {"release_id": release_id, "registry_version": str(version.version), "model_name": MODEL_NAME,
              "verified": True, "verification": verified, "mode": "historical_replay"}
    write_json(output, record)
    return record
