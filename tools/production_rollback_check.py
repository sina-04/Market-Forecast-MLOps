"""Rehearse rollback to a retained verified release, then restore the current release."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from production_acceptance import credentials, request
from production_operations_check import ROOT, command, wait_ready


def deploy(image):
    command(sys.executable, "deploy/pin_release.py", "--image", image)
    command("docker", "compose", "run", "--rm", "register")
    command("docker", "compose", "up", "-d", "--no-build", "--force-recreate", "--wait", "api")
    wait_ready("http://127.0.0.1:8000")


def main(output):
    history = json.loads((ROOT / "deploy/local/deployment.json").read_text())
    current = history[-1]
    previous = next((entry for entry in reversed(history[:-1]) if entry["release_id"] != current["release_id"]), None)
    if previous is None:
        raise SystemExit("Rollback rehearsal requires a retained, previously registered release")
    for entry in [current, previous]:
        image_id = command("docker", "image", "inspect", entry["image"], "--format", "{{.Id}}").strip()
        assert image_id == entry["image_id"], "Retained image identity changed"
    key = credentials()["MARKETFORECAST_API_KEY"]
    url = "http://127.0.0.1:8000"
    timestamp = request(url, "/v1/replay/timestamps", key)["timestamps"][0]["as_of"]
    original = request(url, "/v1/predict", key, {"as_of": timestamp})
    try:
        deploy(previous["image"])
        restored = request(url, "/v1/predict", key, {"as_of": timestamp})
        assert restored["release_id"] == previous["release_id"]
        assert restored["predicted_adjusted_price"] == original["predicted_adjusted_price"]
    finally:
        deploy(current["image"])
    final = request(url, "/v1/predict", key, {"as_of": timestamp})
    assert final["release_id"] == current["release_id"]
    assert final["registry_version"] == original["registry_version"]
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "rolled_back_to": previous["release_id"],
              "restored_current_release": current["release_id"], "current_registry_version": final["registry_version"],
              "known_prediction_reproduced": True, "accepted": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/production/rollback.json")
    main(parser.parse_args().output)
