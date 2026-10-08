"""Pinned VibeSys analysis, plus clearly separated overlap-safe timeline measurements."""

import argparse
import gzip
import io
import json
import math
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from ._vendor import nsys, torch
from .models import Metric
from .store import EvidenceStore

_LOCK = threading.RLock()
PROVENANCE = json.loads((Path(__file__).parent / "_vendor/provenance.json").read_text())


def _report(module, command, args):
    """Retain the upstream human/agent report exactly, without global stdout redirection."""
    stream = io.StringIO()
    previous = module._print

    def output(*values, sep=" ", end="\n", file=None, flush=False):
        if file is None:
            stream.write(sep.join(map(str, values)) + end)

    module._print = output
    connections = []
    if module is nsys:
        opener = module._open_db

        def tracked(path):
            connection, strings = opener(path)
            connections.append(connection)
            return connection, strings

        module._open_db = tracked
    try:
        command(args)
        return stream.getvalue()
    finally:
        module._print = previous
        if module is nsys:
            module._open_db = opener
            for connection in connections:
                connection.close()


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _timeline(events):
    groups = {}
    for event in events:
        if event.get("cat") not in torch._KERNEL_CATS or event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        # Never merge intervals belonging to different GPUs into a utilization estimate.
        device = str(args.get("device", event.get("pid", "unknown")))
        groups.setdefault(device, []).append((event["ts"], event["ts"] + event.get("dur", 0)))
    result = []
    for device, intervals in sorted(groups.items()):
        merged = _union(intervals)
        span = merged[-1][1] - merged[0][0]
        busy = sum(end - start for start, end in merged)
        result.append(
            {
                "device": device,
                "kernel_span_us": span,
                "kernel_union_busy_us": busy,
                "internal_gap_us": span - busy,
                "scope": "between first and last observed kernel; excludes trace edges",
            }
        )
    return result


def _torch_analysis(path):
    with gzip.open(path, "rt") if str(path).endswith(".gz") else path.open() as file:
        raw = json.load(file)
    if not isinstance(raw, dict) or not isinstance(raw.get("traceEvents"), list):
        raise ValueError("expected a raw PyTorch/Kineto Chrome trace with traceEvents")
    for event in raw["traceEvents"]:
        if not isinstance(event, dict):
            raise ValueError("invalid trace event")
        if event.get("ph") == "X":
            for key in ("ts", "dur"):
                value = event.get(key, 0)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("trace timestamp/duration must be finite")
            if event.get("dur", 0) < 0:
                raise ValueError("trace duration cannot be negative")
    index = torch._index_trace(raw)
    correlations = torch._build_op_to_kernels(index)
    certification = [asdict(item) for item in torch._certify(index, correlations)]
    summary = torch._summarize_chrome_trace(raw)
    args = argparse.Namespace(
        report=str(path), trace=str(path), top=15, peak_tflops=None, peak_gbps=None, device=None
    )
    reports = {"summary": _report(torch, torch.cmd_summary, args)}
    shapes = [asdict(shape) for shape in torch._extract_gemm_shapes(index, correlations)]
    reports["gemm_shapes"] = _report(
        torch, torch.cmd_gemm_shapes, argparse.Namespace(trace=str(path), top=15, out=None)
    )
    missing = []
    roofline = []
    try:
        peak_tflops, peak_gbps, _ = torch._resolve_peaks(args, raw)
        roofline = [
            asdict(row)
            for row in torch._extract_roofline_rows(index, correlations, peak_tflops, peak_gbps)
        ]
        reports["roofline"] = _report(torch, torch.cmd_roofline, args)
    except SystemExit:
        missing.append("roofline unavailable: device peaks not identifiable from trace")
    metrics = [
        Metric(name, summary[name], "us", "vibesys:torch").record()
        for name in ("total_cuda_time_us", "total_cpu_time_us")
    ]
    if not index.kernels:
        missing.append("no GPU kernel events; GPU bottleneck attribution unsupported")
    if any(item["status"] == "FAIL" for item in certification):
        missing.append("VibeSys trace certification has FAIL checks; inspect certification")
    return {
        "metrics": metrics,
        "summary": summary,
        "certification": certification,
        "gemm_shapes": shapes,
        "roofline": roofline,
        "reports": reports,
        "timeline": _timeline(raw["traceEvents"]),
        "limitations": missing,
        "status": "observed" if index.kernels else "unsupported",
    }


def _nsys_analysis(path):
    uri = path.resolve().as_uri() + "?mode=ro"
    metrics = []
    timeline = []
    missing = []
    with sqlite3.connect(uri, uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            missing.append("no CUPTI kernel table; GPU bottleneck attribution unsupported")
        else:
            count, total = db.execute(
                "SELECT COUNT(*),SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()
            metrics.extend(
                [
                    Metric("kernel_count", count, "calls", "vibesys:nsys").record(),
                    Metric(
                        "kernel_duration_sum",
                        total / 1000 if total is not None else None,
                        "us",
                        "vibesys:nsys",
                        unavailable_reason="no kernels" if total is None else None,
                    ).record(),
                ]
            )
            column = nsys._kernel_name_col(db)
            if column and nsys._column_exists(db, "CUPTI_ACTIVITY_KIND_KERNEL", "deviceId"):
                query = f'SELECT "{column}",start,end,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'
                rows = db.execute(query).fetchall()
                devices = sorted({row[3] for row in rows})
                for device in devices:
                    kernels = [row for row in rows if row[3] == device]
                    gaps, busy, idle = nsys._device_gaps(kernels, {})
                    timeline.append(
                        {
                            "device": str(device),
                            "kernel_union_busy_us": busy / 1000,
                            "internal_gap_over_1us_us": idle / 1000,
                            "gap_count_over_1us": len(gaps),
                            "scope": "VibeSys device_gaps threshold >1us; excludes edges",
                        }
                    )
        for table in (
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        ):
            if table not in tables:
                missing.append("unavailable table: " + table)
    args = argparse.Namespace(report=str(path), top=15, step=1)
    commands = {
        "kernels": nsys.cmd_kernels,
        "cpu_overhead": nsys.cmd_cpu_overhead,
        "idle_gaps": nsys.cmd_idle_gaps,
        "memory": nsys.cmd_memory,
        "graph_replays": nsys.cmd_graph_replays,
        "step_timeline": nsys.cmd_step_timeline,
    }
    reports = {}
    for name, command in commands.items():
        try:
            reports[name] = _report(nsys, command, args)
        except (sqlite3.Error, ValueError, RuntimeError) as exc:
            missing.append(f"{name} unavailable: {type(exc).__name__}")
    observed = any(m["name"] == "kernel_count" and m["value"] for m in metrics)
    return {
        "metrics": metrics,
        "reports": reports,
        "timeline": timeline,
        "limitations": missing,
        "status": "observed" if observed else "unsupported",
    }


def analyze_trace(path: str | Path, profiler: str):
    path = Path(path)
    with _LOCK:
        if profiler == "torch":
            return _torch_analysis(path)
        if profiler == "nsys":
            return _nsys_analysis(path)
        raise ValueError("profiler must be torch or nsys")


def import_trace(
    store: EvidenceStore,
    identity,
    path: str | Path,
    *,
    profiler: str,
    worker: str,
    capture_id: str | None = None,
):
    """Analyze a stable owned copy and store an explicitly worker-labelled profile."""
    if worker not in identity.workers:
        raise ValueError("worker is not declared in deployment identity")
    path = Path(path)
    raw_artifact = store.put_artifact(path)
    # Exported artifacts use content-addressed names. Retain a public format hint,
    # not the original filename, so downstream analyzers can restore required suffixes.
    raw_artifact["format"] = (
        "nsys_rep"
        if path.suffix == ".nsys-rep"
        else "chrome_trace_json_gzip"
        if path.name.endswith(".gz")
        else "nsys_sqlite"
        if profiler == "nsys"
        else "chrome_trace_json"
    )
    artifacts = [raw_artifact]
    stored = store.root / raw_artifact["path"]
    # Format suffix affects gzip parsing. Analyze in an isolated temporary directory.
    with tempfile.TemporaryDirectory() as temp:
        copy = Path(temp) / (
            "trace.nsys-rep"
            if path.suffix == ".nsys-rep"
            else "trace.json.gz"
            if path.name.endswith(".gz")
            else "trace.json"
        )
        shutil.copyfile(stored, copy)
        if path.suffix == ".nsys-rep":
            if profiler != "nsys":
                raise ValueError(".nsys-rep requires the nsys profiler")
            nsys_exe = shutil.which("nsys")
            if not nsys_exe:
                raise RuntimeError(
                    "nsys CLI is required to export .nsys-rep; import SQLite instead"
                )
            sqlite_path = Path(temp) / "trace.sqlite"
            try:
                subprocess.run(
                    [nsys_exe, "export", "--type=sqlite", f"--output={sqlite_path}", str(copy)],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError("Nsight export failed: " + type(exc).__name__) from exc
            exported = store.put_artifact(sqlite_path)
            exported["format"] = "nsys_sqlite"
            artifacts.append(exported)
            copy = sqlite_path
        try:
            result = analyze_trace(copy, profiler)
        except Exception as exc:
            store.event(
                "trace_analysis_failed",
                {"artifact": raw_artifact["digest"], "error": type(exc).__name__},
            )
            raise
    record = {
        "id": uuid.uuid4().hex,
        "timestamp": time.time(),
        "capture_id": capture_id,
        "worker": worker,
        "profiler": profiler,
        "source": "vibesys_profiler",
        "diagnostic": True,
        "correctness": "not_evaluated",
        "provenance": PROVENANCE,
        "artifacts": artifacts,
        **result,
        "time_semantics": "Kernel duration sums include overlap; timeline union is separate. "
        "Profiler results are diagnostic, not unprofiled serving latency.",
    }
    store.profile(identity, record)
    return record
