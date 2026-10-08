"""Versioned durable SQLite captures, readable while the collector is running."""

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .collector import Scrape
from .config import GPU, Config
from .metrics import Sample


class Store:
    def __init__(self, path, *, readonly=False):
        self.path = Path(path)
        if readonly:
            self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version != 1:
                self.connection.close()
                raise ValueError(f"Unsupported capture schema {version}")
            return
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self.connection.close()
            raise ValueError(f"Unsupported capture schema {version}")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, created_ns INTEGER NOT NULL,
                manifest TEXT NOT NULL, config_sha256 TEXT NOT NULL, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frames (
                run_id TEXT NOT NULL REFERENCES runs(id), seq INTEGER NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY(run_id, seq)
            );
            PRAGMA user_version=1;
        """)

    def start(self, config, *, run_id=None):
        run_id = run_id or str(uuid.uuid4())
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    time.time_ns(),
                    json.dumps(config.manifest()),
                    config.sha256,
                    "collecting",
                ),
            )
        return run_id

    def append(self, run_id, seq, scrapes):
        payload = json.dumps([s.record() for s in scrapes], allow_nan=False)
        with self.connection:
            self.connection.execute("INSERT INTO frames VALUES (?, ?, ?)", (run_id, seq, payload))

    def finish(self, run_id, status="complete"):
        if status not in {"complete", "interrupted", "failed"}:
            raise ValueError("Invalid completion status")
        with self.connection:
            self.connection.execute("UPDATE runs SET status=? WHERE id=?", (status, run_id))

    def latest(self):
        row = self.connection.execute(
            "SELECT id FROM runs ORDER BY created_ns DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("Capture has no runs")
        return row[0]

    def runs(self, limit=50):
        return [
            {"id": row[0], "created_ns": row[1], "status": row[2]}
            for row in self.connection.execute(
                "SELECT id, created_ns, status FROM runs ORDER BY created_ns DESC LIMIT ?", (limit,)
            )
        ]

    def read(self, run_id=None):
        run_id = run_id or self.latest()
        row = self.connection.execute(
            "SELECT manifest, status, config_sha256 FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown run {run_id}")
        manifest = json.loads(row[0])
        manifest["gpus"] = tuple(GPU(**g) for g in manifest["gpus"])
        config = Config(**manifest)
        if config.sha256 != row[2]:
            raise ValueError("Capture manifest hash mismatch")
        frames = []
        for _, payload in self.connection.execute(
            "SELECT seq, payload FROM frames WHERE run_id=? ORDER BY seq", (run_id,)
        ):
            frame = []
            for item in json.loads(payload):
                item["samples"] = tuple(Sample(**s) for s in item["samples"])
                frame.append(Scrape(**item))
            frames.append(tuple(frame))
        return config, frames, {"run_id": run_id, "status": row[1], "config_sha256": row[2]}

    def export_jsonl(self, destination, run_id=None):
        run_id = run_id or self.latest()
        config, frames, identity = self.read(run_id)
        with Path(destination).open("w") as stream:
            stream.write(
                json.dumps({"schema_version": 1, **identity, "manifest": config.manifest()}) + "\n"
            )
            for seq, frame in enumerate(frames):
                stream.write(
                    json.dumps({"seq": seq, "scrapes": [s.record() for s in frame]}) + "\n"
                )

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
