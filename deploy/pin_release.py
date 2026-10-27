"""Give a verified local image its content-derived release tag and retain deployment history."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", default="marketforecast-service:local")
args = parser.parse_args()
release = json.loads(subprocess.check_output(["docker", "run", "--rm", "--entrypoint", "cat", args.image,
                                             "/bundle/release.json"], text=True))
tag = "marketforecast-service:" + release["release_id"]
subprocess.run(["docker", "tag", args.image, tag], check=True)
image_id = subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip()
path = root / ".env"
if not path.is_file():
    raise SystemExit("Run deploy/setup.py first")
lines = [line for line in path.read_text().splitlines() if not line.startswith("MARKETFORECAST_IMAGE=")]
path.write_text("\n".join(lines + ["MARKETFORECAST_IMAGE=" + tag]) + "\n", encoding="utf-8")
directory = root / "deploy/local"
directory.mkdir(parents=True, exist_ok=True)
manifest_path = directory / "deployment.json"
history = json.loads(manifest_path.read_text()) if manifest_path.is_file() else []
record = {"image": tag, "image_id": image_id, "release_id": release["release_id"],
          "pinned_at_utc": datetime.now(timezone.utc).isoformat()}
if not history or history[-1]["image_id"] != image_id:
    history.append(record)
manifest_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
