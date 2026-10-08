import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from obsrv import Config, Store, analyze, collect_once, load_config
from obsrv.cli import main
from obsrv.config import GPU
from obsrv.demo import demo_frames
from obsrv.metrics import GPU_METRICS, Sample, counter_delta, gauge, parse_metrics
from obsrv.trace import trace_gaps


@pytest.fixture
def config():
    return Config(
        "worker",
        "vllm",
        "http://localhost:8000/metrics",
        "http://localhost:9400/metrics",
        (GPU("gpu-0", {"UUID": "GPU-DEMO"}),),
        allocation="exclusive",
    )


def test_demo_conserves_idle_and_counts_outage_as_uncovered(config):
    report = analyze(config, demo_frames(config))
    row = report["gpus"][0]
    assert row["window_seconds"] == 9
    assert row["covered_seconds"] == 7
    assert row["estimated_engine_idle_seconds"] == pytest.approx(4.36)
    assert row["estimated_engine_idle_seconds"] == pytest.approx(
        row["idle_seconds_no_demand"]
        + row["idle_seconds_with_demand"]
        + row["idle_seconds_unknown_demand"]
    )
    assert row["low_activity_with_demand_seconds"] == 1
    assert report["scrape_errors"]["gpu"] == 1
    assert report["serving"]["generation_tokens"]["rate_per_second"] == 30
    assert report["serving"]["ttft"]["mean_seconds"] == pytest.approx(0.3)


def test_time_weighting_is_not_a_sample_average(config):
    frames = demo_frames(config)[:3]
    for i, (seconds, active) in enumerate(zip((1, 2, 4), (0, 0.25, 0.75), strict=True)):
        start = seconds * 1_000_000_000
        frames[i] = tuple(
            replace(s, started_ns=start, ended_ns=start, wall_time_ns=start) for s in frames[i]
        )
        frames[i] = (
            replace(
                frames[i][0],
                samples=(Sample("DCGM_FI_PROF_GR_ENGINE_ACTIVE", {"UUID": "GPU-DEMO"}, active),),
            ),
            frames[i][1],
        )
    row = analyze(config, frames)["gpus"][0]
    assert row["covered_seconds"] == 3
    assert row["estimated_engine_idle_seconds"] == pytest.approx(1.25)


def test_missing_engine_activity_does_not_infer_idle_from_sm_or_gpu_util(config):
    frames = [
        tuple(
            replace(
                s, samples=tuple(x for x in s.samples if x.name != "DCGM_FI_PROF_GR_ENGINE_ACTIVE")
            )
            for s in frame
        )
        for frame in demo_frames(config)
    ]
    row = analyze(config, frames)["gpus"][0]
    assert row["covered_seconds"] == 0
    assert row["estimated_engine_idle_seconds"] is None


@pytest.mark.parametrize(
    "text",
    [
        'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} NaN',
        'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} +Inf',
        'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} -1',
        'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} 9.223372036854776e18',
        'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} 2',
        'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} 0 1000',
    ],
)
def test_invalid_or_stale_gpu_values_are_unknown(text):
    samples = parse_metrics(text, GPU_METRICS, now_seconds=100, max_age_seconds=3)
    assert gauge(samples, GPU_METRICS["engine_active"], {"UUID": "GPU-DEMO"}, ratio=True) is None


def test_parser_handles_escapes_and_ignores_nonallowlisted_metrics():
    text = r'DCGM_FI_PROF_SM_ACTIVE{UUID="GPU-X",pod="a\"b"} 0.5' + "\n"
    text += "private_request_prompt 123\n"
    samples = parse_metrics(text, GPU_METRICS, 100, 3)
    assert len(samples) == 1
    assert samples[0].labels["pod"] == 'a"b'


def test_duplicate_tp_gauges_are_unknown_until_selected():
    samples = [Sample("vllm:num_requests_running", {"rank": str(i)}, 4) for i in range(2)]
    assert gauge(samples, ("vllm:num_requests_running",), {}) is None
    assert gauge(samples, ("vllm:num_requests_running",), {"rank": "0"}) == 4


@pytest.mark.parametrize("old,new", [(10, 2), (None, 12), (10, None)])
def test_counter_resets_and_invalid_samples_are_not_work(old, new):
    assert (
        counter_delta([Sample("c_total", {}, old)], [Sample("c_total", {}, new)], ("c_total",), {})
        is None
    )


def test_counter_series_churn_and_duplicate_series_are_unknown():
    old = [Sample("c_total", {"rank": "0"}, 10)]
    new = [Sample("c_total", {"rank": "1"}, 12)]
    assert counter_delta(old, new, ("c_total",), {}) is None
    assert counter_delta(old, old * 2, ("c_total",), {}) is None


def test_long_gap_is_uncovered(config):
    frames = demo_frames(config)[:2]
    frames[1] = tuple(
        replace(s, started_ns=s.started_ns + 10_000_000_000, ended_ns=s.ended_ns + 10_000_000_000)
        for s in frames[1]
    )
    row = analyze(config, frames)["gpus"][0]
    assert row["window_seconds"] == 11
    assert row["coverage_fraction"] == 0
    assert row["idle_seconds_no_demand"] is None


def test_endpoint_skew_does_not_attribute_idle_to_demand(config):
    frames = demo_frames(config)[:2]
    frames = [
        (
            gpu,
            replace(
                serving,
                started_ns=serving.started_ns + 600_000_000,
                ended_ns=serving.ended_ns + 600_000_000,
            ),
        )
        for gpu, serving in frames
    ]
    row = analyze(config, frames)["gpus"][0]
    assert row["estimated_engine_idle_seconds"] == 1
    assert row["idle_seconds_unknown_demand"] == 1
    assert row["demand_classified_seconds"] == 0


def test_empty_capture_cannot_claim_idle(config):
    row = analyze(config, [])["gpus"][0]
    assert row["coverage_fraction"] == 0
    assert row["estimated_engine_idle_seconds"] is None


def test_sqlite_roundtrip_and_multiple_runs(config, tmp_path):
    with Store(tmp_path / "capture.sqlite") as store:
        first = store.start(config)
        for seq, frame in enumerate(demo_frames(config)):
            store.append(first, seq, frame)
        store.finish(first)
        second = store.start(replace(config, metadata={"candidate_revision": "next"}))
        loaded, frames, identity = store.read(first)
        assert loaded == config
        assert analyze(loaded, frames) == analyze(config, demo_frames(config))
        assert identity["status"] == "complete"
        assert store.latest() == second
        store.export_jsonl(tmp_path / "samples.jsonl", first)
    lines = [json.loads(x) for x in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert lines[0]["schema_version"] == 1
    assert lines[0]["run_id"] == first
    assert len(lines) == 11


def test_manifest_tampering_is_detected(config, tmp_path):
    with Store(tmp_path / "capture.sqlite") as store:
        run = store.start(config)
        store.connection.execute("UPDATE runs SET config_sha256='bad' WHERE id=?", (run,))
        with pytest.raises(ValueError, match="hash mismatch"):
            store.read(run)


def test_config_rejects_physical_and_mig_double_counting(config):
    with pytest.raises(ValueError, match="MIG"):
        replace(config, gpus=(config.gpus[0], GPU("mig", {"UUID": "GPU-DEMO", "GPU_I_ID": "1"})))


def test_distinct_mig_instances_are_supported(config):
    cfg = replace(
        config, gpus=tuple(GPU(str(i), {"UUID": "GPU-DEMO", "GPU_I_ID": str(i)}) for i in (1, 2))
    )
    assert len(cfg.gpus) == 2


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/metrics",
        "http://user:pw@localhost/metrics",
        "http://localhost/metrics?token=secret",
    ],
)
def test_config_does_not_persist_url_credentials(config, url):
    with pytest.raises(ValueError):
        replace(config, serving_url=url)


def test_toml_examples_and_unknown_keys(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert load_config(root / "examples/vllm.toml").engine == "vllm"
    assert load_config(root / "examples/sglang.toml").engine == "sglang"
    bad = tmp_path / "bad.toml"
    bad.write_text('typo = "yes"\n')
    with pytest.raises(ValueError, match="Unknown"):
        load_config(bad)


def test_collector_records_endpoint_failure_without_false_idle(config):
    def failed(url, timeout):
        raise OSError("sensitive data must not be persisted")

    frames = [collect_once(config, fetcher=failed)]
    assert all(s.error == "OSError" and not s.samples for s in frames[0])
    assert analyze(config, frames)["gpus"][0]["estimated_engine_idle_seconds"] is None


def test_real_http_collect_store_report(config, tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            text = (
                'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} 0.25\n'
                if self.path == "/gpu"
                else "vllm:num_requests_running 2\nvllm:num_requests_waiting 1\n"
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(text.encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    cfg = replace(config, serving_url=base + "/serving", dcgm_url=base + "/gpu")
    try:
        captured = collect_once(cfg)
        assert all(s.error is None for s in captured)
        # Deterministic timestamps for interval math; HTTP transport is real.
        frames = [
            tuple(
                replace(s, started_ns=i * 1_000_000_000, ended_ns=i * 1_000_000_000)
                for s in captured
            )
            for i in (1, 2)
        ]
        with Store(tmp_path / "capture.sqlite") as store:
            run = store.start(cfg)
            for seq, frame in enumerate(frames):
                store.append(run, seq, frame)
            store.finish(run)
        out = tmp_path / "report.json"
        assert main(["report", "--db", str(tmp_path / "capture.sqlite"), "--out", str(out)]) == 0
        row = json.loads(out.read_text())["gpus"][0]
        assert row["idle_seconds_with_demand"] == 0.75
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_sglang_adapter(config):
    cfg = replace(config, engine="sglang")

    def fake(url, timeout):
        if url == cfg.dcgm_url:
            return 'DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-DEMO"} 0.2\n'
        return "sglang:num_running_reqs 3\nsglang:num_queue_reqs 1\n"

    scrapes = collect_once(cfg, fetcher=fake)
    frames = [
        tuple(replace(s, started_ns=i * 1_000_000_000, ended_ns=i * 1_000_000_000) for s in scrapes)
        for i in (1, 2)
    ]
    assert analyze(cfg, frames)["gpus"][0]["idle_seconds_with_demand"] == 0.8


def test_trace_unions_overlap_and_excludes_cpu_other_device_and_clips():
    def event(cat, ts, dur, device=0):
        return {"ph": "X", "cat": cat, "ts": ts, "dur": dur, "args": {"device": device}}

    document = {
        "traceEvents": [
            event("kernel", -2, 6),
            event("gpu_memcpy", 2, 4),
            event("kernel", 8, 1),
            event("cpu_op", 0, 10),
            event("kernel", 0, 10, 1),
        ]
    }
    result = trace_gaps(document, device=0, start_us=0, end_us=10)
    assert result["gaps_us"] == [[6, 8], [9, 10]]
    assert result["gap_fraction"] == pytest.approx(0.3)
    assert result["event_counts"]["kernel"] == 2


def test_empty_trace_is_not_100_percent_idle():
    with pytest.raises(ValueError, match="No GPU events"):
        trace_gaps({"traceEvents": []}, device=0, start_us=0, end_us=10)


def test_demo_cli_generates_readable_artifacts(tmp_path):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["metadata"]["data_kind"] == "synthetic"
    assert report["scrape_errors"]["gpu"] == 1
