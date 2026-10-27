"""Read-only historical inference; target prices are never used as inputs."""
from datetime import datetime
from pathlib import Path
import threading
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from marketforecast.data import validate_snapshot
from marketforecast.features import engineer
from marketforecast.hourly import clean_hourly
from marketforecast.io import read_json

TEHRAN = ZoneInfo("Asia/Tehran")


def local_timestamp(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("as_of must include a timezone offset")
    stamp = stamp.astimezone(TEHRAN)
    if stamp.microsecond:
        raise ValueError("as_of must identify an exact bar timestamp")
    return stamp.replace(tzinfo=None).isoformat(sep=" ")


def aware_timestamp(value):
    return datetime.fromisoformat(str(value)).replace(tzinfo=TEHRAN).isoformat()


def prepare_features(root):
    snapshot = Path(root) / "snapshot"
    validate_snapshot(snapshot)
    raw = pd.read_csv(snapshot / "raw.csv")
    cleaned, _, _ = clean_hourly(raw, snapshot)
    frame, names, _ = engineer(cleaned)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame, names


class Predictor:
    def __init__(self, root):
        from .bundle import validate_bundle
        self.root = Path(root)
        self.release = validate_bundle(root)
        self.metadata = read_json(self.root / "model_metadata.json")
        self.names = self.metadata["feature_order"]
        if (self.metadata["selected"]["length"] != 60 or len(self.names) != 49
                or self.metadata["selected"]["units"] != 32
                or self.metadata["frequency"] != "hourly" or self.metadata["target_mode"] != "log_return"
                or self.names != read_json(self.root / "feature_order.json")):
            raise ValueError("Unsupported or inconsistent hourly model contract")
        frame, names = prepare_features(root)
        if names != self.names:
            raise ValueError("Feature order differs from saved model")
        self.index = pd.read_csv(self.root / "replay_index.csv")
        if len(self.index) != 535 or self.index.as_of.duplicated().any():
            raise ValueError("Expected exactly 535 unique verified replay windows")
        self.scalers = joblib.load(self.root / "scalers.joblib")
        if self.scalers["features"].n_features_in_ != 49 or self.scalers["target"].n_features_in_ != 1:
            raise ValueError("Scaler dimensions disagree with model")
        values = frame[self.names].to_numpy(float)
        scaled = self.scalers["features"].transform(values)
        positions = self.index.position.to_numpy(int)
        if np.any(positions < 60) or np.any(positions >= len(frame)):
            raise ValueError("Replay positions outside available history")
        for entry in self.index.itertuples():
            window = frame.iloc[entry.position - 60:entry.position + 1]
            if (not np.all(np.diff(window.session_number) == 1) or window.segment.nunique() != 1
                    or window.date.iloc[-2] != entry.as_of or window.date.iloc[-1] != entry.target_timestamp):
                raise ValueError("Replay window crosses missing data or differs from its timestamps")
        self.windows = np.stack([scaled[t-60:t] for t in positions]).astype("float32")
        self.previous = frame.adj_hourly_last_price.to_numpy(float)[positions - 1]
        self.end_features = values[positions - 1]
        self.lookup = {entry.as_of: i for i, entry in enumerate(self.index.itertuples())}
        if not np.isfinite(self.windows).all() or not np.isfinite(self.previous).all():
            raise ValueError("Nonfinite inference inputs")
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        self.model = tf.keras.models.load_model(self.root / "model.keras", compile=False, safe_mode=True)
        if tuple(self.model.input_shape[1:]) != (60, 49) or self.model.output_shape[-1] != 1:
            raise ValueError("Saved model has incompatible dimensions")
        self.lock = threading.Lock()
        self._predict(self.windows[:1], self.previous[:1])

    def _predict(self, windows, previous):
        with self.lock:
            values = self.model(windows, training=False).numpy()
        log_return = self.scalers["target"].inverse_transform(values).ravel()
        with np.errstate(over="ignore", invalid="ignore"):
            prices = np.asarray(previous) * np.exp(log_return)
        if not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("Model produced an invalid price")
        return prices

    def predict_batch(self):
        return np.concatenate([self._predict(self.windows[i:i+128], self.previous[i:i+128])
                               for i in range(0, len(self.windows), 128)])

    def predict(self, as_of):
        normalized = local_timestamp(as_of)
        if normalized not in self.lookup:
            raise KeyError("Timestamp is not an eligible historical test window")
        i = self.lookup[normalized]
        row = self.index.iloc[i]
        return {"input_timestamp": aware_timestamp(row.as_of),
                "target_timestamp": aware_timestamp(row.target_timestamp),
                "predicted_adjusted_price": float(self._predict(self.windows[i:i+1], self.previous[i:i+1])[0]),
                "persistence_forecast": float(self.previous[i])}

    def timestamps(self):
        return [{"as_of": aware_timestamp(row.as_of), "target_timestamp": aware_timestamp(row.target_timestamp)}
                for row in self.index.itertuples()]
