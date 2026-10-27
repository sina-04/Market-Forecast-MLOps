"""Real HTTP parity, monitoring and paced load checks; standard-library only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def credentials(path=".env"):
    values = {}
    if Path(path).is_file():
        values.update(line.split("=", 1) for line in Path(path).read_text().splitlines()
                      if "=" in line and not line.startswith("#"))
    values.update({key: value for key, value in os.environ.items() if key in ["MARKETFORECAST_API_KEY", "GRAFANA_ADMIN_PASSWORD"]})
    return values


def request(url, path, key=None, body=None):
    headers = {"X-API-Key": key} if key else {}
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        content = response.read().decode()
        return json.loads(content) if "application/json" in response.headers.get("Content-Type", "") else content


def run(url, key, expected_path, duration):
    started = datetime.now(timezone.utc).isoformat()
    model = request(url, "/v1/model", key)
    entries = request(url, "/v1/replay/timestamps", key)["timestamps"]
    expected = list(csv.DictReader(Path(expected_path).open(encoding="utf-8")))
    assert len(entries) == len(expected) == 535
    maximum = 0
    for entry, saved in zip(entries, expected):
        result = request(url, "/v1/predict", key, {"as_of": entry["as_of"]})
        observed, reference = result["predicted_adjusted_price"], float(saved["lstm"])
        assert abs(observed - reference) <= .001 + 1e-6 * abs(reference), (entry, observed, reference)
        assert datetime.fromisoformat(result["target_timestamp"]).replace(tzinfo=None).isoformat(sep=" ") == saved["date"]
        maximum = max(maximum, abs(observed - reference))
    for body, status in [({"as_of": "invalid"}, 422), ({"as_of": "2000-01-01T09:00:00+03:30"}, 404)]:
        try:
            request(url, "/v1/predict", key, body)
            raise AssertionError("Invalid input accepted")
        except urllib.error.HTTPError as error:
            assert error.code == status
    try:
        request(url, "/v1/model")
        raise AssertionError("Unauthenticated access accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 401

    def measured(index):
        before = time.perf_counter()
        try:
            request(url, "/v1/predict", key, {"as_of": entries[index % len(entries)]["as_of"]})
            return time.perf_counter() - before, None
        except Exception as error:
            return time.perf_counter() - before, type(error).__name__

    for i in range(10):
        assert measured(i)[1] is None
    with ThreadPoolExecutor(max_workers=5) as executor:
        begin = time.perf_counter()
        futures = []
        for i in range(duration * 5):
            delay = begin + i / 5 - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(executor.submit(measured, i))
        values = [future.result() for future in futures]
    times = sorted(value[0] for value in values)
    failures = [value[1] for value in values if value[1]]
    p95 = times[math.ceil(.95 * len(times)) - 1]
    metric_text = request(url, "/metrics")
    def metric(name):
        return float(next(line.split()[-1] for line in metric_text.splitlines() if line.startswith(name + " ")))
    peak = metric("forecast_container_memory_peak_bytes")
    return {"verified_at_utc": started, "release_id": model["release_id"], "registry_version": model["registry_version"],
            "mode": model["mode"], "http_parity": {"predictions": 535, "max_absolute_difference": maximum,
            "rtol": 1e-6, "atol_adjusted_rials": .001},
            "load": {"requests": len(values), "rate_per_second": 5, "duration_seconds": duration,
                     "concurrency_limit": 5, "unexpected_errors": len(failures), "p95_seconds": p95,
                     "container_peak_bytes": peak},
            "accepted": not failures and p95 < .5 and 0 < peak < 2 * 1024**3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected", default="results/foolad-hourly-a100/training-results/hourly-forecast/test_predictions.csv")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("artifacts/production/acceptance.json"))
    args = parser.parse_args()
    if args.duration < 1:
        parser.error("duration must be positive")
    result = run(args.url, credentials()["MARKETFORECAST_API_KEY"], args.expected, args.duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)
