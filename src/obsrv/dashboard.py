"""Loopback-only dashboard for retained captures and externally produced benchmarks."""

import json
import sqlite3
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from .analysis import analyze
from .benchmarks import read_benchmarks
from .local import local_sample
from .store import Store


def demo_benchmarks():
    from .benchmarks import paired_benchmark

    result = []
    for iteration, speedup in enumerate((1.0, 1.18, 1.36, 1.31)):
        result.append(
            paired_benchmark(
                {
                    "mode": "benchmark",
                    "passed": True,
                    "data_kind": "synthetic",
                    "scope": "Illustrative paired replay; no measured performance claim",
                    "device": {"device_name": "Illustrative deployment"},
                    "records": [
                        {
                            "case": case,
                            "reference_s": latency,
                            "candidate_s": latency / speedup,
                            "output_tokens": 120,
                        }
                        for case, latency in (("edit", 2.4), ("test", 2.7), ("repair", 2.5))
                    ],
                },
                name=(
                    "Baseline",
                    "Batch formation",
                    "Reuse unchanged spans",
                    "Rejected experiment",
                )[iteration],
                observed_ns=iteration * 1_000_000_000,
            )
        )
    return result


class Dashboard:
    def __init__(
        self, *, databases=(), benchmark_dir=None, demo=False, local_monitor=False, journal=None
    ):
        if demo and (databases or benchmark_dir or local_monitor or journal):
            raise ValueError("Synthetic demo cannot be mixed with real data sources")
        self.databases = tuple(Path(p) for p in databases)
        self.benchmark_dir = Path(benchmark_dir) if benchmark_dir else None
        self.demo = demo
        self.local_monitor = local_monitor
        self.journal = Path(journal) if journal else None
        self.history = deque(maxlen=300)
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def monitor(self):
        while not self.stop.is_set():
            sample = local_sample()
            with self.lock:
                self.history.append(sample)
            if self.journal:
                try:
                    self.journal.parent.mkdir(parents=True, exist_ok=True)
                    # Bound the journal at 10 MiB; keep one rotated file.
                    if self.journal.exists() and self.journal.stat().st_size > 10 * 1024 * 1024:
                        self.journal.replace(self.journal.with_suffix(".previous.jsonl"))
                    with self.journal.open("a") as stream:
                        stream.write(json.dumps(sample, allow_nan=False) + "\n")
                except OSError:
                    with self.lock:
                        sample["journal_status"] = "write_failed"
            self.stop.wait(2)

    def state(self):
        reports, issues, benchmarks = [], [], []
        for database in self.databases:
            try:
                with Store(database, readonly=True) as store:
                    for run in reversed(store.runs(limit=20)):
                        config, frames, identity = store.read(run["id"])
                        reports.append(
                            {**analyze(config, frames), **identity, "created_ns": run["created_ns"]}
                        )
            except (ValueError, OSError, sqlite3.Error, TypeError, KeyError):
                issues.append({"kind": "capture_unavailable", "name": database.name})
        if self.benchmark_dir:
            benchmarks, benchmark_issues = read_benchmarks(self.benchmark_dir)
            issues.extend(benchmark_issues)
        if self.demo:
            from .config import GPU, Config
            from .demo import demo_frames

            for name, factor in (("Baseline deployment", 1.0), ("Candidate deployment", 0.5)):
                config = Config(
                    name,
                    "vllm",
                    "http://localhost:8000/metrics",
                    "http://localhost:9400/metrics",
                    (GPU("GPU 0", {"UUID": "GPU-DEMO"}),),
                    allocation="exclusive",
                    metadata={"data_kind": "synthetic"},
                )
                frames = demo_frames(config)
                report = analyze(config, frames)
                # The two demo deployments are distinct synthetic captures, generated below.
                if factor != 1:
                    from dataclasses import replace

                    frames = [
                        (
                            replace(
                                g,
                                samples=tuple(
                                    replace(s, value=1 - (1 - s.value) * factor)
                                    if s.name == "DCGM_FI_PROF_GR_ENGINE_ACTIVE"
                                    and s.value is not None
                                    else s
                                    for s in g.samples
                                ),
                            ),
                            serving,
                        )
                        for g, serving in frames
                    ]
                    report = analyze(config, frames)
                reports.append(
                    {
                        **report,
                        "run_id": name,
                        "status": "synthetic",
                        "created_ns": len(reports) * 1_000_000_000,
                    }
                )
            benchmarks = demo_benchmarks()
        with self.lock:
            local = list(self.history)
        return {
            "schema_version": 1,
            "updated_ns": time.time_ns(),
            "mode": "synthetic_demo" if self.demo else "local_evidence",
            "reports": reports,
            "benchmarks": benchmarks,
            "local": local,
            "issues": issues,
            "monitor_enabled": self.local_monitor,
        }


def serve(
    *, port=8765, databases=(), benchmark_dir=None, demo=False, local_monitor=False, journal=None
):
    if not 0 < port < 65536:
        raise ValueError("port must be in 1..65535")
    app = Dashboard(
        databases=databases,
        benchmark_dir=benchmark_dir,
        demo=demo,
        local_monitor=local_monitor,
        journal=journal,
    )
    assets = files("obsrv").joinpath("web")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?", 1)[0]
            if route == "/api/state":
                body = json.dumps(app.state(), allow_nan=False).encode()
                content_type = "application/json"
            elif route in {"/", "/app.js", "/style.css"}:
                filename = "index.html" if route == "/" else route[1:]
                body = assets.joinpath(filename).read_bytes()
                content_type = {
                    "index.html": "text/html",
                    "app.js": "text/javascript",
                    "style.css": "text/css",
                }[filename]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; "
                "style-src 'self'; script-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = None
    if local_monitor:
        thread = threading.Thread(target=app.monitor, daemon=True)
        thread.start()
    print(f"obsrv dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop.set()
        server.server_close()
        if thread:
            thread.join(timeout=4)
