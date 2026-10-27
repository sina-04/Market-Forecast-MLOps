"""Exercise registry independence, restart persistence, backup, isolated restore and monitoring."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from production_acceptance import credentials, request

ROOT = Path(__file__).resolve().parents[1]


def command(*arguments):
    process = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"Command failed: {' '.join(arguments)}\n{process.stdout[-2000:]}\n{process.stderr[-2000:]}")
    return process.stdout


def wait_ready(url):
    for _ in range(90):
        try:
            request(url, "/health/ready")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1)
    raise RuntimeError("API readiness timed out")


def audit_count(compose):
    return int(command(*compose, "exec", "-T", "api", "python", "-c",
                       "from marketforecast.service.audit import AuditStore; print(AuditStore('/state/audit.sqlite').count())").strip())


def main(output):
    key = credentials()["MARKETFORECAST_API_KEY"]
    compose = ["docker", "compose"]
    url = "http://127.0.0.1:8000"
    wait_ready(url)
    model = request(url, "/v1/model", key)
    entry = request(url, "/v1/replay/timestamps", key)["timestamps"][0]
    expected = request(url, "/v1/predict", key, {"as_of": entry["as_of"]})
    before = audit_count(compose)
    command(*compose, "restart", "api")
    wait_ready(url)
    after = audit_count(compose)
    assert after >= before
    assert request(url, "/v1/model", key)["release_id"] == model["release_id"]
    command(*compose, "stop", "mlflow")
    try:
        independent = request(url, "/v1/predict", key, {"as_of": entry["as_of"]})
        assert independent["predicted_adjusted_price"] == expected["predicted_adjusted_price"]
    finally:
        command(*compose, "start", "mlflow")

    # Repeated registration must preserve the current immutable model version.
    command(*compose, "run", "--rm", "register")
    registration = json.loads(command(*compose, "exec", "-T", "api", "cat", "/registry/registration.json"))
    assert registration["registry_version"] == model["registry_version"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_name = "acceptance-" + stamp
    command(*compose, "stop", "mlflow")
    try:
        command(*compose, "run", "--rm", "--no-deps", "backup", "marketforecast-service", "backup",
                "--output", "/backups/" + backup_name)
    finally:
        command(*compose, "start", "mlflow")
    command(sys.executable, "deploy/export_backup.py", "--name", backup_name)
    project = "marketforecast-recovery-" + stamp
    recovery = ["docker", "compose", "-p", project, "-f", "compose.yaml", "-f", "deploy/recovery.compose.yaml"]
    try:
        command(*recovery, "run", "--rm", "--no-deps", "restore", "marketforecast-service", "restore",
                "--input", "/backups/" + backup_name)
        command(*recovery, "up", "-d", "--no-build", "--wait", "mlflow", "api")
        restored_count = audit_count(recovery)
        assert restored_count >= after
        restored = request("http://127.0.0.1:8001", "/v1/predict", key, {"as_of": entry["as_of"]})
        assert restored["release_id"] == model["release_id"]
        assert restored["registry_version"] == model["registry_version"]
        assert restored["predicted_adjusted_price"] == expected["predicted_adjusted_price"]
        # Verify the restored registry database and artifact-backed model version.
        registry = command(*recovery, "exec", "-T", "mlflow", "python", "-c",
            "from mlflow import MlflowClient; c=MlflowClient(tracking_uri='http://localhost:5000'); "
            "print(c.get_model_version_by_alias('foolad-hourly-lstm-replay','champion').version)")
        assert registry.strip() == model["registry_version"]
    finally:
        command(*recovery, "down")

    targets = request("http://127.0.0.1:9090", "/api/v1/targets")["data"]["activeTargets"]
    assert any(target["labels"].get("job") == "forecast" and target["health"] == "up" for target in targets)
    rules = request("http://127.0.0.1:9090", "/api/v1/rules")["data"]["groups"]
    assert sum(len(group["rules"]) for group in rules) == 5
    auth = base64.b64encode(("admin:" + credentials()["GRAFANA_ADMIN_PASSWORD"]).encode()).decode()
    req = urllib.request.Request("http://127.0.0.1:3000/api/dashboards/uid/forecast-replay", headers={"Authorization": "Basic " + auth})
    with urllib.request.urlopen(req) as response:
        dashboard = json.load(response)
    assert len(dashboard["dashboard"]["panels"]) == 9
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "release_id": model["release_id"],
              "registry_version": model["registry_version"], "restart_persisted_records": after,
              "registry_offline_inference": True, "registration_idempotent": True,
              "backup": "deploy/local/backups/" + backup_name,
              "restored_records": restored_count, "restored_prediction_and_registry": True,
              "recovery_project": project, "prometheus_target_up": True, "alert_rules": 5,
              "grafana_panels": 9, "accepted": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/production/operations.json")
    main(parser.parse_args().output)
