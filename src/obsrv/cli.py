"""Headless operator interface. Collection never triggers profiles."""

import argparse
import json
import sys
import time

from .activation import activate, deployment_directory, deployment_settings
from .analysis import import_trace
from .collector import Collector
from .config import Config
from .evidence import export_context, report, verify_export
from .profiling import ProfileController
from .store import EvidenceStore, StorageLimitError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="obsrv", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "profile", "import", "report", "export-context"):
        child = sub.add_parser(name)
        selection = child.add_mutually_exclusive_group(required=True)
        selection.add_argument("--config")
        selection.add_argument(
            "--deployment", help="deployment activated in this working directory"
        )
        if name == "collect":
            child.add_argument("--count", type=int, help="finite scrape count; default continuous")
        elif name == "profile":
            child.add_argument("--duration", type=float, default=30)
            child.add_argument("--recover", metavar="CAPTURE_ID")
        elif name == "import":
            child.add_argument("path")
            child.add_argument("--profiler", choices=("torch", "nsys"), required=True)
            child.add_argument("--worker", required=True)
        elif name == "report":
            child.add_argument("--all-versions", action="store_true")
    child = sub.add_parser("activate", help="discover a deployment and start passive collection")
    child.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    child.add_argument("--url", required=True)
    child.add_argument("--deployment")
    child.add_argument("--revision", default="unknown")
    child.add_argument("--model")
    child.add_argument("--state-dir")
    child.add_argument("--trace-dir")
    child.add_argument("--auth-env")
    child.add_argument("--count", type=int)
    child.add_argument("--setup-only", action="store_true")
    child = sub.add_parser(
        "deployment-settings", help="print engine launch settings; do not launch"
    )
    child.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    child.add_argument("--trace-dir", required=True)
    child = sub.add_parser("verify", help="verify a self-contained exported bundle")
    child.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "deployment-settings":
        print(json.dumps(deployment_settings(args.engine, args.trace_dir), indent=2))
        return 0
    if args.command == "verify":
        verify_export(args.path)
        print("Export integrity verified")
        return 0
    if args.command == "activate":
        config, path, guidance = activate(
            engine=args.engine,
            url=args.url,
            deployment=args.deployment,
            revision=args.revision,
            model=args.model,
            state_dir=args.state_dir,
            trace_dir=args.trace_dir,
            auth_env=args.auth_env,
        )
        print(
            json.dumps(
                {
                    "config": str(path),
                    **guidance,
                    "next_profile": ["obsrv", "profile", "--config", str(path)],
                    "next_export": ["obsrv", "export-context", "--config", str(path)],
                },
                indent=2,
            ),
            flush=True,
        )
        if args.setup_only:
            return 0
        args.config = str(path)
        args.command = "collect"
    else:
        if args.deployment:
            args.config = str(deployment_directory(args.deployment) / "obsrv.json")
        config = Config.load(args.config)
    store = EvidenceStore(
        config.state_dir, retention_days=config.retention_days, max_bytes=config.max_storage_bytes
    )
    try:
        if args.command == "collect":
            if args.count is not None and args.count < 1:
                parser.error("--count must be positive")
            collector = Collector(config, store)
            done = 0
            while args.count is None or done < args.count:
                # Reload explicit rollout identity; never infer code versions from metrics.
                collector.reconfigure(Config.load(args.config))
                record = collector.collect_once()
                print(json.dumps(record, allow_nan=False), flush=True)
                done += 1
                if args.count is None or done < args.count:
                    time.sleep(collector.config.interval_seconds)
        elif args.command == "profile":
            controller = ProfileController(config, store)
            result = (
                controller.recover(args.recover)
                if args.recover
                else controller.capture(duration_seconds=args.duration)
            )
            print(json.dumps(result, indent=2))
            return 0 if result["state"] in {"completed", "recovered"} else 2
        elif args.command == "import":
            print(
                json.dumps(
                    import_trace(
                        store,
                        config.identity,
                        args.path,
                        profiler=args.profiler,
                        worker=args.worker,
                    ),
                    indent=2,
                )
            )
        elif args.command == "report":
            print(
                json.dumps(report(store, None if args.all_versions else config.identity), indent=2)
            )
        elif args.command == "export-context":
            print(export_context(store, config.identity))
        return 0
    finally:
        store.close()


def entrypoint():
    try:
        return main()
    except KeyboardInterrupt:
        return 130
    except (ValueError, TypeError, FileNotFoundError, StorageLimitError) as exc:
        print(f"obsrv: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Transport exceptions can contain credentials and bodies. Emit only the error class.
        print(f"obsrv failed: {type(exc).__name__}; inspect local status/report", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
