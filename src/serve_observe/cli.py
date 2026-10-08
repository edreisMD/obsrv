"""Local collection and agent evidence exports; no inference proxy or deployment mutations."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from .analysis import analyze
from .collector import collect_once
from .config import load_config
from .store import Store


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Record DCGM and serving metrics in SQLite")
    collect.add_argument("--config", required=True)
    collect.add_argument("--db", required=True)
    collect.add_argument("--duration", type=float, default=60)
    report = commands.add_parser(
        "report", help="Export an evidence report for an optimization agent"
    )
    report.add_argument("--db", required=True)
    report.add_argument("--run")
    report.add_argument("--out", required=True)
    export = commands.add_parser("export", help="Export stored samples as versioned JSONL")
    export.add_argument("--db", required=True)
    export.add_argument("--run")
    export.add_argument("--out", required=True)
    demo = commands.add_parser(
        "demo", help="Generate deterministic synthetic evidence without a GPU"
    )
    demo.add_argument("--out", default="artifacts/demo")
    trace = commands.add_parser("trace", help="Analyze observed GPU gaps in a Chrome/PyTorch trace")
    trace.add_argument("--input", required=True)
    trace.add_argument("--device", required=True)
    trace.add_argument("--start-us", type=float, required=True)
    trace.add_argument("--end-us", type=float, required=True)
    trace.add_argument("--out", required=True)
    dashboard = commands.add_parser(
        "dashboard", help="View capture comparisons and optimization history"
    )
    dashboard.add_argument("--db", action="append", default=[])
    dashboard.add_argument("--benchmark-dir")
    dashboard.add_argument("--demo", action="store_true")
    dashboard.add_argument("--monitor-local", action="store_true")
    dashboard.add_argument("--journal", help="Local Apple telemetry JSONL journal")
    dashboard.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "dashboard":
            from .dashboard import serve

            if args.demo and (args.db or args.benchmark_dir or args.monitor_local):
                raise ValueError("Demo mode cannot be mixed with real captures or local monitoring")
            serve(
                port=args.port,
                databases=args.db,
                benchmark_dir=args.benchmark_dir,
                demo=args.demo,
                local_monitor=args.monitor_local,
                journal=args.journal,
            )
            return 0
        if args.command == "trace":
            from .trace import trace_gaps

            document = json.loads(Path(args.input).read_text())
            _write_json(
                args.out,
                trace_gaps(
                    document, device=args.device, start_us=args.start_us, end_us=args.end_us
                ),
            )
            return 0
        if args.command == "demo":
            from .demo import make_demo

            print(json.dumps(make_demo(Path(args.out))))
            return 0
        if args.command == "collect":
            if not math.isfinite(args.duration) or args.duration <= 0:
                raise ValueError("duration must be finite and positive")
            config = load_config(args.config)
            with Store(args.db) as store:
                run = store.start(config)
                print(json.dumps({"run_id": run, "config_sha256": config.sha256}), flush=True)
                deadline = time.monotonic() + args.duration
                next_scrape = time.monotonic()
                seq = 0
                try:
                    while time.monotonic() < deadline:
                        store.append(run, seq, collect_once(config))
                        seq += 1
                        next_scrape += config.interval_seconds
                        # Missed ticks are skipped, never followed by a catch-up burst.
                        if next_scrape < time.monotonic():
                            next_scrape = time.monotonic() + config.interval_seconds
                        time.sleep(max(0, min(next_scrape, deadline) - time.monotonic()))
                except KeyboardInterrupt:
                    store.finish(run, "interrupted")
                    return 130
                except Exception:
                    store.finish(run, "failed")
                    raise
                store.finish(run)
            return 0
        with Store(args.db) as store:
            if args.command == "export":
                store.export_jsonl(args.out, args.run)
            else:
                config, frames, identity = store.read(args.run)
                _write_json(args.out, {**analyze(config, frames), **identity})
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"serve-observe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
