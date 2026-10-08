"""Durable observations, capture ownership and content-addressed artifacts."""

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from .models import Identity, canonical, digest


class StorageLimitError(RuntimeError):
    pass


class EvidenceStore:
    def __init__(self, root: str | Path, *, retention_days=7.0, max_bytes=5 * 1024**3):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        self.exports = self.root / "exports"
        self.exports.mkdir(exist_ok=True)
        self.retention_seconds = retention_days * 86400
        self.max_bytes = max_bytes
        self._last_prune = 0.0
        self.db = sqlite3.connect(self.root / "obsrv.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS identities(id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(
          id INTEGER PRIMARY KEY, identity_id TEXT NOT NULL REFERENCES identities(id),
          timestamp REAL NOT NULL, body TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS observation_identity_time
          ON observations(identity_id,timestamp);
        CREATE TABLE IF NOT EXISTS captures(
          id TEXT PRIMARY KEY, identity_id TEXT NOT NULL REFERENCES identities(id),
          deployment TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
          state TEXT NOT NULL, cleanup_confirmed INTEGER NOT NULL, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS profiles(
          id TEXT PRIMARY KEY, identity_id TEXT NOT NULL REFERENCES identities(id),
          timestamp REAL NOT NULL, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def register(self, identity: Identity):
        self.db.execute(
            "INSERT OR IGNORE INTO identities VALUES(?,?)",
            (identity.fingerprint, canonical(identity.record())),
        )
        self.db.commit()
        return identity.fingerprint

    def event(self, kind, details):
        self.db.execute(
            "INSERT INTO events(timestamp,kind,body) VALUES(?,?,?)",
            (
                time.time(),
                kind,
                canonical({"event_kind": kind, "recorded_at": time.time(), **details}),
            ),
        )
        self.db.commit()

    def usage(self):
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def ensure_capacity(self, incoming=0):
        if time.monotonic() - self._last_prune > 60:
            self.prune()
            self._last_prune = time.monotonic()
        if self.usage() + incoming > self.max_bytes:
            # A bounded tiny control log can survive a full data quota; don't retain payloads.
            self.event("storage_limit", {"incoming_bytes": incoming, "max_bytes": self.max_bytes})
            self.db.execute(
                "DELETE FROM events WHERE id NOT IN (SELECT id FROM events "
                "ORDER BY id DESC LIMIT 100)"
            )
            self.db.commit()
            raise StorageLimitError("storage quota reached; collection/import paused")

    def prune(self, *, now=None):
        cutoff = (time.time() if now is None else now) - self.retention_seconds
        self.db.execute("DELETE FROM observations WHERE timestamp < ?", (cutoff,))
        self.db.execute("DELETE FROM profiles WHERE timestamp < ?", (cutoff,))
        self.db.execute(
            "DELETE FROM captures WHERE updated < ? AND cleanup_confirmed=1 "
            "AND state IN ('completed','partial','failed','cancelled','recovered')",
            (cutoff,),
        )
        self.db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        self.db.commit()
        # Artifacts pinned by profiles or exported manifests are preserved.
        referenced = set()
        for (body,) in self.db.execute("SELECT body FROM profiles"):
            for artifact in json.loads(body).get("artifacts", []):
                referenced.add(artifact["digest"]["value"])
        for p in self.exports.glob("*/bundle.json"):
            bundle = json.loads(p.read_text())
            for artifact in bundle.get("artifacts", []):
                referenced.add(artifact["digest"]["value"])
        for p in self.artifacts.iterdir():
            if p.is_file() and p.name not in referenced and p.stat().st_mtime < cutoff:
                p.unlink()
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def observation(self, identity, record):
        self.ensure_capacity(len(canonical(record).encode()) + 4096)
        key = self.register(identity)
        self.db.execute(
            "INSERT INTO observations(identity_id,timestamp,body) VALUES(?,?,?)",
            (key, record["end"], canonical(record)),
        )
        self.db.commit()

    def rows(self, table, identity_id=None):
        if table not in {"observations", "profiles", "captures", "identities", "events"}:
            raise ValueError("invalid evidence table")
        query = f"SELECT body FROM {table}"
        args = ()
        if identity_id is not None:
            query += " WHERE identity_id=?"
            args = (identity_id,)
        return [json.loads(row[0]) for row in self.db.execute(query, args)]

    def put_artifact(self, source: Path):
        # Copy before hashing; never retain a hash of one file version and bytes of another.
        self.ensure_capacity(source.stat().st_size)
        fd, temp = tempfile.mkstemp(dir=self.artifacts, prefix=".import-")
        try:
            sha = hashlib.sha256()
            with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
                while block := src.read(1024 * 1024):
                    sha.update(block)
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            checksum = sha.hexdigest()
            target = self.artifacts / checksum
            if target.exists():
                if self.hash_file(target) != checksum:
                    raise ValueError("stored artifact digest mismatch")
                Path(temp).unlink()
            else:
                os.rename(temp, target)
                target.chmod(0o444)
            return {
                "path": f"artifacts/{checksum}",
                "digest": {"algorithm": "sha256", "value": checksum},
                "size_bytes": target.stat().st_size,
            }
        finally:
            Path(temp).unlink(missing_ok=True)

    @staticmethod
    def hash_file(path):
        sha = hashlib.sha256()
        with Path(path).open("rb") as file:
            while block := file.read(1024 * 1024):
                sha.update(block)
        return sha.hexdigest()

    def profile(self, identity, record):
        self.ensure_capacity(len(canonical(record).encode()) + 4096)
        key = self.register(identity)
        self.db.execute(
            "INSERT INTO profiles VALUES(?,?,?,?)",
            (record["id"], key, record["timestamp"], canonical(record)),
        )
        self.db.commit()

    def save_capture(self, identity, session):
        key = self.register(identity)
        session["updated"] = time.time()
        self.db.execute(
            "INSERT OR REPLACE INTO captures VALUES(?,?,?,?,?,?,?,?)",
            (
                session["id"],
                key,
                identity.deployment,
                session["created"],
                session["updated"],
                session["state"],
                int(session["cleanup_confirmed"]),
                canonical(session),
            ),
        )
        self.db.commit()

    def outstanding(self, deployment):
        return [
            json.loads(row[0])
            for row in self.db.execute(
                "SELECT body FROM captures WHERE deployment=? AND cleanup_confirmed=0",
                (deployment,),
            )
        ]

    def diagnostic_overlap(self, deployment, start, end):
        rows = self.db.execute(
            "SELECT body FROM captures WHERE deployment=? AND created<=?", (deployment, end)
        )
        return any(json.loads(r[0]).get("finished", float("inf")) >= start for r in rows)

    @contextlib.contextmanager
    def capture_lock(self, deployment):
        locks = self.root / "locks"
        locks.mkdir(exist_ok=True)
        with (locks / digest(deployment)).open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another obsrv process owns this deployment capture") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def save_export(self, bundle, summary):
        content = canonical(bundle)
        key = digest(bundle)
        target = self.exports / key
        if target.exists():
            return target
        self.ensure_capacity(
            len(content.encode())
            + len(summary.encode())
            + sum(a["size_bytes"] for a in bundle["artifacts"])
        )
        temporary = Path(tempfile.mkdtemp(dir=self.exports, prefix=".export-"))
        try:
            (temporary / "artifacts").mkdir()
            for artifact in bundle["artifacts"]:
                source = self.root / artifact["path"]
                if self.hash_file(source) != artifact["digest"]["value"]:
                    raise ValueError("artifact digest mismatch during export")
                shutil.copyfile(source, temporary / artifact["path"])
            (temporary / "bundle.json").write_text(content + "\n")
            (temporary / "context.md").write_text(summary)
            (temporary / "manifest.json").write_text(
                canonical(
                    {
                        "bundle_sha256": self.hash_file(temporary / "bundle.json"),
                        "context_sha256": self.hash_file(temporary / "context.md"),
                        "artifacts": bundle["artifacts"],
                    }
                )
                + "\n"
            )
            os.rename(temporary, target)
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
