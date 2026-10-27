"""Export the baked, immutable bundle to a local checksummed ZIP without running the API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image")
args = parser.parse_args()
values = dict(line.split("=", 1) for line in (root / ".env").read_text().splitlines() if "=" in line)
image = args.image or values["MARKETFORECAST_IMAGE"]
manifest = json.loads(subprocess.check_output(["docker", "run", "--rm", "--entrypoint", "cat", image,
                                              "/bundle/release.json"], text=True))
directory = root / "artifacts/production/releases" / manifest["release_id"]
if not directory.exists():
    directory.parent.mkdir(parents=True, exist_ok=True)
    container = subprocess.check_output(["docker", "create", "--entrypoint", "true", image], text=True).strip()
    try:
        subprocess.run(["docker", "cp", container + ":/bundle", str(directory)], check=True, capture_output=True)
    finally:
        subprocess.run(["docker", "rm", container], check=True, capture_output=True)
for name, expected in manifest["checksums"].items():
    actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("Export checksum failed: " + name)
archive = directory.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stream.write(path, path.relative_to(directory))
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n", encoding="utf-8")
print(json.dumps({"release_id": manifest["release_id"], "archive": str(archive), "sha256": digest,
                  "files_verified": len(manifest["checksums"])}, indent=2))
