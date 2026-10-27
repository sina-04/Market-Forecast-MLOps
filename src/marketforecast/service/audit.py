"""Durable audit records and deduplicated replay evaluation inputs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


class AuditStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                prediction_id TEXT PRIMARY KEY, release_id TEXT NOT NULL,
                received_at TEXT NOT NULL, as_of TEXT, target_timestamp TEXT,
                status INTEGER NOT NULL, latency_seconds REAL NOT NULL,
                prediction REAL, persistence REAL, registry_version TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS replay_targets ON requests(release_id, target_timestamp)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def ready(self):
        # A write transaction checks more than a SELECT against read-only storage.
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE requests SET status=status WHERE 0")
        return True

    def record(self, prediction_id, release_id, status, latency, as_of=None, result=None, registry_version=None):
        result = result or {}
        with self.connection() as db:
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?)", (
                prediction_id, release_id, datetime.now(timezone.utc).isoformat(), as_of,
                result.get("target_timestamp"), status, latency, result.get("predicted_adjusted_price"),
                result.get("persistence_forecast"), registry_version))

    def latest(self, release_id, limit=100):
        with self.connection() as db:
            # Pick the latest recorded request per target, then the most recent target dates.
            rows = db.execute("""SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY target_timestamp ORDER BY received_at DESC) AS rank
                FROM requests WHERE release_id=? AND status=200
            ) WHERE rank=1 ORDER BY target_timestamp DESC LIMIT ?""", (release_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def count(self):
        with self.connection() as db:
            return db.execute("SELECT count(*) FROM requests").fetchone()[0]

    def backup(self, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as source:
            target = sqlite3.connect(destination)
            try:
                source.backup(target)
            finally:
                target.close()
        return destination
