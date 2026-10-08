"""Operator-only native engine capture. Durable ownership survives collector restarts."""

import math
import threading
import time
import urllib.error
import uuid
from pathlib import Path

from .analysis import import_trace
from .config import Config
from .http import HttpTransport, ProfileRejectedError, Transport
from .models import digest
from .store import EvidenceStore


class ProfileController:
    def __init__(self, config: Config, store: EvidenceStore, transport: Transport | None = None):
        self.config = config
        self.store = store
        self.transport = transport or HttpTransport(config.auth_env)

    def _post(self, operation, body, timeout):
        self.transport.post(self.config.serving_url.rstrip("/") + "/" + operation, body, timeout)

    def _stop(self, session):
        session["state"] = "stopping"
        self.store.save_capture(self.config.identity, session)
        try:
            self._post("stop_profile", {}, self.config.flush_timeout_seconds)
        except Exception as exc:
            session["state"] = "cleanup_required"
            session["error"] = "stop/export failed: " + type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                session["http_status"] = exc.code
            self.store.save_capture(self.config.identity, session)
            return False
        session["cleanup_confirmed"] = True
        session["finished"] = time.time()
        return True

    def recover(self, session_id: str):
        """Explicitly stop an outstanding obsrv-owned session after operator verification."""
        with self.store.capture_lock(self.config.identity.deployment):
            sessions = self.store.outstanding(self.config.identity.deployment)
            session = next((s for s in sessions if s["id"] == session_id), None)
            if session is None:
                raise ValueError("no outstanding capture with that ID")
            if session["identity_id"] != self.config.identity.fingerprint or (
                session["endpoint_id"] != digest(self.config.serving_url)
            ):
                raise ValueError("recovery config must match original deployment and endpoint")
            if self._stop(session):
                session["state"] = "recovered"
                session.setdefault("limitations", []).append(
                    "recovered capture; import available traces explicitly"
                )
                self.store.save_capture(self.config.identity, session)
            return session

    def capture(self, *, duration_seconds=30.0, cancel: threading.Event | None = None):
        if not math.isfinite(duration_seconds) or not 0 < duration_seconds <= 300:
            raise ValueError("active capture duration must be >0 and <=300 seconds")
        config = self.config
        if len(config.identity.workers) > 1 and not config.trace_patterns:
            raise ValueError("multi-worker capture requires trace_patterns for worker attribution")
        cancel = cancel or threading.Event()
        with self.store.capture_lock(config.identity.deployment):
            if self.store.outstanding(config.identity.deployment):
                raise RuntimeError("outstanding capture requires explicit recovery before starting")
            self.store.ensure_capacity()
            session = {
                "id": uuid.uuid4().hex,
                "created": time.time(),
                "state": "starting",
                "cleanup_confirmed": False,
                "identity_id": config.identity.fingerprint,
                "endpoint_id": digest(config.serving_url),
                "duration_seconds": duration_seconds,
                "profiler": "torch",
                "profiles": [],
                "limitations": [],
            }
            config.trace_dir.mkdir(parents=True, exist_ok=True)
            before = {
                p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in config.trace_dir.rglob("*")
                if p.is_file()
            }
            trace_root = config.trace_dir
            body = {}
            if config.identity.engine == "sglang":
                trace_root = config.trace_dir / session["id"]
                trace_root.mkdir()
                server_root = Path(config.server_trace_dir or str(config.trace_dir.resolve()))
                body = {
                    "output_dir": str(server_root / session["id"]),
                    "activities": ["CPU", "GPU"],
                    "merge_profiles": False,
                    "record_shapes": True,
                    "with_stack": False,
                }
            self.store.save_capture(config.identity, session)
            error = None
            started = False
            try:
                self._post("start_profile", body, config.request_timeout_seconds)
                started = True
                session["state"] = "recording"
                session["recording_started"] = time.time()
                self.store.save_capture(config.identity, session)
                cancel.wait(duration_seconds)
            except BaseException as exc:
                error = exc
                # A definitive rejected start must not stop someone else's profiler.
                if isinstance(exc, ProfileRejectedError) or (
                    isinstance(exc, urllib.error.HTTPError)
                    and 400 <= exc.code < 500
                    and exc.code not in {408, 429}
                ):
                    session["cleanup_confirmed"] = True
                    session["finished"] = time.time()
            finally:
                if not session["cleanup_confirmed"]:
                    self._stop(session)
                if error:
                    session["error"] = "capture interrupted: " + type(error).__name__
                    session["guidance"] = (
                        "Check native profiling setup, endpoint access and server logs."
                    )
                    if isinstance(error, urllib.error.HTTPError):
                        session["http_status"] = error.code
                if not session["cleanup_confirmed"]:
                    session["state"] = "cleanup_required"
                elif error or not started:
                    session["state"] = "failed"
                elif cancel.is_set():
                    session["state"] = "cancelled"
                else:
                    session["state"] = "flushed"
                self.store.save_capture(config.identity, session)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error
            if session["state"] in {"failed", "cleanup_required"}:
                return session
            patterns = config.trace_patterns or {config.identity.workers[0]: "**/*"}
            candidates = {}
            for worker, pattern in patterns.items():
                for path in trace_root.glob(pattern):
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or not (path.name.endswith(".json") or path.name.endswith(".json.gz"))
                    ):
                        continue
                    if before.get(path.resolve()) == (path.stat().st_size, path.stat().st_mtime_ns):
                        continue
                    candidates.setdefault(path.resolve(), []).append(worker)
            observed = set()
            for path, workers in candidates.items():
                if len(workers) != 1:
                    session["limitations"].append("ambiguous trace worker mapping; trace skipped")
                    continue
                try:
                    profile = import_trace(
                        self.store,
                        config.identity,
                        path,
                        profiler="torch",
                        worker=workers[0],
                        capture_id=session["id"],
                    )
                    session["profiles"].append(profile["id"])
                    if profile["status"] == "observed":
                        observed.add(workers[0])
                except Exception as exc:
                    session["limitations"].append("trace import failed: " + type(exc).__name__)
            session["missing_workers"] = sorted(set(config.identity.workers) - observed)
            if session["missing_workers"]:
                session["limitations"].append(
                    "missing valid GPU profiles for some declared workers"
                )
            if not config.identity.configuration.get("topology_verified", True):
                session["limitations"].append(
                    "worker topology unverified; no complete worker coverage claim"
                )
            if session["state"] != "cancelled":
                session["state"] = "partial" if session["limitations"] else "completed"
            self.store.save_capture(config.identity, session)
            return session
