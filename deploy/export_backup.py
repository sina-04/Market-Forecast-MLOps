"""Export a named backup from Docker storage with the invoking host user's ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", required=True)
parser.add_argument("--project", default="marketforecast")
args = parser.parse_args()
if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name) or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.project):
    parser.error("Use simple backup and Compose project names")
values = dict(line.split("=", 1) for line in (root / ".env").read_text().splitlines() if "=" in line)
image = values.get("MARKETFORECAST_IMAGE", "marketforecast-service:local")
destination = root / "deploy/local/backups" / args.name
if destination.exists():
    raise SystemExit("Backup export already exists; choose a fresh name")
destination.parent.mkdir(parents=True, exist_ok=True)
container = subprocess.check_output(["docker", "create", "--mount",
    f"type=volume,source={args.project}_backup_data,target=/backups,readonly", "--entrypoint", "true", image], text=True).strip()
try:
    subprocess.run(["docker", "cp", container + ":/backups/" + args.name, str(destination)], check=True, capture_output=True)
finally:
    subprocess.run(["docker", "rm", container], check=True, capture_output=True)
checksums = json.loads((destination / "checksums.json").read_text())
for name, expected in checksums.items():
    path = (destination / name).resolve()
    if not path.is_relative_to(destination.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Exported backup checksum mismatch: " + name)
print(json.dumps({"backup": str(destination), "files_verified": len(checksums)}, indent=2))
