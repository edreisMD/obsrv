import argparse
import importlib.util
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from obsrv import analyze_trace, import_trace
from obsrv._vendor import nsys, torch
from obsrv.analysis import PROVENANCE, _report


def test_torch_exact_measurements_and_overlap(trace, raw_trace):
    result = analyze_trace(trace, "torch")
    assert result["summary"] == torch._summarize_chrome_trace(raw_trace)
    assert result["summary"]["total_cuda_time_us"] == 120
    assert result["timeline"][0]["kernel_union_busy_us"] == 90
    assert result["timeline"][0]["internal_gap_us"] == 20
    assert result["gemm_shapes"][0]["m"] == 4
    assert result["gemm_shapes"][0]["total_gpu_time_us"] == 50
    args = argparse.Namespace(report=str(trace), top=15)
    assert result["reports"]["summary"] == _report(torch, torch.cmd_summary, args)
    assert "accepted_round" not in result


def test_parity_with_original_vibesys_analyzer(trace, raw_trace):
    reference = os.environ.get("VIBESYS_REFERENCE_ROOT")
    if not reference:
        pytest.skip("optional VIBESYS_REFERENCE_ROOT is not set")
    path = Path(reference) / "resources/profilers/torch/analyze_torch_profile.py"
    if not path.exists():
        pytest.skip("source checkout absent; pinned vendor parity tested separately")
    spec = importlib.util.spec_from_file_location("original_vibesys_torch", path)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    result = analyze_trace(trace, "torch")
    assert result["summary"] == original._summarize_chrome_trace(raw_trace)
    index = original._index_trace(raw_trace)
    assert result["certification"] == [
        asdict(c) for c in original._certify(index, original._build_op_to_kernels(index))
    ]
    args = argparse.Namespace(report=str(trace), top=15)
    assert result["reports"]["summary"] == _report(original, original.cmd_summary, args)


def create_nsys(path):
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
        INSERT INTO StringIds VALUES(1,'gemm'),(2,'cudaLaunchKernel');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
          shortName INTEGER,start INTEGER,end INTEGER,deviceId INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
          (1,0,50000,0,1),(1,20000,70000,0,2),(1,90000,110000,0,3);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
          nameId INTEGER,start INTEGER,end INTEGER,correlationId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(2,0,1000,1);
        """)


def test_nsys_reports_and_device_union_match_reference(tmp_path):
    path = tmp_path / "nsys.sqlite"
    create_nsys(path)
    result = analyze_trace(path, "nsys")
    assert result["status"] == "observed"
    assert result["timeline"][0]["kernel_union_busy_us"] == 90
    assert result["timeline"][0]["internal_gap_over_1us_us"] == 20
    assert result["reports"]["kernels"] == _report(
        nsys, nsys.cmd_kernels, argparse.Namespace(report=str(path), top=15)
    )
    assert "Avg CPU launch: 1.0 us" in result["reports"]["cpu_overhead"]
    assert any("GRAPH_TRACE" in missing for missing in result["limitations"])


def test_import_hashes_and_missing_gpu(config, store, trace, tmp_path):
    profile = import_trace(store, config.identity, trace, profiler="torch", worker="rank0")
    assert profile["provenance"] == PROVENANCE
    artifact = profile["artifacts"][0]
    assert store.hash_file(store.root / artifact["path"]) == artifact["digest"]["value"]
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"traceEvents": []}))
    assert (
        import_trace(store, config.identity, empty, profiler="torch", worker="rank0")["status"]
        == "unsupported"
    )


def test_corrupt_trace_records_failure_without_payload(store, config, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json secret contents")
    with pytest.raises(ValueError):
        import_trace(store, config.identity, path, profiler="torch", worker="rank0")
    assert store.rows("profiles") == []
    assert "secret" not in str(store.rows("events"))


def test_nsys_parity_with_original_source(tmp_path):
    reference = os.environ.get("VIBESYS_REFERENCE_ROOT")
    if not reference:
        pytest.skip("optional VIBESYS_REFERENCE_ROOT is not set")
    source = Path(reference) / "resources/profilers/nsys/analyze_nsys.py"
    if not source.exists():
        pytest.skip("original VibeSys checkout unavailable")
    spec = importlib.util.spec_from_file_location("original_vibesys_nsys", source)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    path = tmp_path / "trace.sqlite"
    create_nsys(path)
    import contextlib
    import io

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        original.cmd_cpu_overhead(argparse.Namespace(report=str(path), top=15))
    result = analyze_trace(path, "nsys")
    assert result["reports"]["cpu_overhead"] == output.getvalue()
    rows = [(1, 0, 50000, 0), (1, 20000, 70000, 0), (1, 90000, 110000, 0)]
    assert original._device_gaps(rows, {}) == nsys._device_gaps(rows, {})
