"""Historical quality and PSI diagnostics. No claim of live concept-drift detection."""
import numpy as np
import pandas as pd
from pathlib import Path
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector, PlatformCollector

from marketforecast.evaluation import metrics
from marketforecast.io import read_json
from .runtime import local_timestamp


def make_reference(frame):
    reference = {}
    for name in frame:
        values = frame[name].to_numpy(float)
        cuts = np.unique(np.quantile(values, np.arange(0.1, 1, 0.1)))
        counts, _ = np.histogram(values, bins=np.r_[-np.inf, cuts, np.inf])
        reference[name] = {"cuts": cuts.tolist(), "probabilities": (counts / len(values)).tolist()}
    return reference


def psi(values, reference):
    counts, _ = np.histogram(values, bins=np.r_[-np.inf, reference["cuts"], np.inf])
    observed = np.maximum(counts / len(values), 1e-6)
    expected = np.maximum(reference["probabilities"], 1e-6)
    observed, expected = observed / observed.sum(), expected / expected.sum()
    return float(np.sum((observed - expected) * np.log(observed / expected)))


class Monitor:
    def __init__(self, predictor, store):
        self.predictor, self.store = predictor, store
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        self.requests = Counter("forecast_requests_total", "API requests", ["route", "status"], registry=self.registry)
        self.latency = Histogram("forecast_request_duration_seconds", "API request latency", ["route"],
                                 buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2, 5), registry=self.registry)
        self.ready = Gauge("forecast_ready", "Model and audit store ready", registry=self.registry)
        self.release = Gauge("forecast_release_info", "Immutable model release", ["release_id"], registry=self.registry)
        self.release.labels(predictor.release["release_id"]).set(1)
        self.samples = Gauge("forecast_replay_samples", "Distinct historical replay targets", registry=self.registry)
        self.quality = Gauge("forecast_replay_error", "Historical replay error", ["model", "metric"], registry=self.registry)
        self.drift = Gauge("forecast_feature_psi", "Historical replay PSI against training", ["feature"], registry=self.registry)
        self.failures = Counter("forecast_evaluation_failures_total", "Background diagnostic failures", registry=self.registry)
        self.container_memory = Gauge("forecast_container_memory_bytes", "Container cgroup memory usage", registry=self.registry)
        self.container_peak = Gauge("forecast_container_memory_peak_bytes", "Container cgroup peak memory usage", registry=self.registry)
        self.container_memory.set_function(lambda: self.cgroup_value("memory.current"))
        self.container_peak.set_function(lambda: self.cgroup_value("memory.peak"))
        self.ready.set(1)
        self.reference = read_json(predictor.root / "reference.json")
        # Labels are loaded only by this evaluator, never by the inference implementation.
        labels = pd.read_csv(predictor.root / "evidence/test_predictions.csv")
        self.actuals = dict(zip(labels.date, labels.actual))

    @staticmethod
    def cgroup_value(name):
        path = Path("/sys/fs/cgroup") / name
        return int(path.read_text()) if path.is_file() else 0

    def evaluate(self):
        rows = self.store.latest(self.predictor.release["release_id"])
        self.samples.set(len(rows))
        self.quality.clear()
        self.drift.clear()
        if len(rows) < 100:
            return {"mode": "historical_replay", "samples": len(rows), "status": "insufficient_samples"}
        actual = [self.actuals[local_timestamp(row["target_timestamp"])] for row in rows]
        report = {"mode": "historical_replay", "samples": len(rows), "status": "evaluated", "metrics": {}, "psi": {}}
        for model, column in [("lstm", "prediction"), ("persistence", "persistence")]:
            scores = metrics(actual, [row[column] for row in rows])
            report["metrics"][model] = scores
            for metric in ["MAE", "RMSE", "MAPE"]:
                self.quality.labels(model, metric).set(scores[metric])
        indices = [self.predictor.lookup[local_timestamp(row["as_of"])] for row in rows]
        for i, name in enumerate(self.predictor.names):
            score = psi(self.predictor.end_features[indices, i], self.reference[name])
            self.drift.labels(name).set(score)
            report["psi"][name] = score
        return report
