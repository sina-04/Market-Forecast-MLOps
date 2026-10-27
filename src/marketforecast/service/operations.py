"""SQLite-consistent backup; invoke with MLflow stopped to freeze its artifact store."""
from pathlib import Path
import shutil
import sqlite3

from marketforecast.io import checksum, write_json
from .audit import AuditStore
from .bundle import verify_files
from marketforecast.io import read_json


def backup(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    AuditStore("/state/audit.sqlite").backup(output / "state/audit.sqlite")
    source = sqlite3.connect("file:/mlflow/mlflow.db?mode=ro", uri=True)
    destination = output / "mlflow/mlflow.db"
    destination.parent.mkdir()
    target = sqlite3.connect(destination)
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    shutil.copytree("/mlflow/artifacts", output / "mlflow/artifacts")
    shutil.copytree("/registry", output / "registry")
    hashes = {str(p.relative_to(output)): checksum(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "checksums.json", hashes)
    return {"backup": str(output), "files": len(hashes),
            "monitoring_history": "Prometheus/Grafana volumes are disposable; provisioned configuration remains in Git"}


def restore(source, state_root="/state", mlflow_root="/mlflow", registry_root="/registry"):
    """Restore to fresh volumes only; never overwrite a running installation."""
    source = Path(source).resolve()
    verify_files(source, read_json(source / "checksums.json"))
    destinations = {"state": Path(state_root), "mlflow": Path(mlflow_root), "registry": Path(registry_root)}
    for name, destination in destinations.items():
        if not (source / name).is_dir():
            raise ValueError(f"Missing backup component: {name}")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Restore requires an empty destination: {destination}")
    for name, destination in destinations.items():
        shutil.copytree(source / name, destination, dirs_exist_ok=True)
    audit = AuditStore(destinations["state"] / "audit.sqlite")
    return {"restored": True, "audit_records": audit.count(), "registration": read_json(destinations["registry"] / "registration.json")}
