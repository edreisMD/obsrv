import json
import math
import plistlib
from types import SimpleNamespace

import pytest

from obsrv.benchmarks import paired_benchmark, read_benchmarks
from obsrv.cli import main
from obsrv.config import GPU, Config
from obsrv.dashboard import Dashboard
from obsrv.demo import demo_frames
from obsrv.local import local_sample
from obsrv.store import Store


def artifact(**overrides):
    return {
        "mode": "benchmark",
        "passed": True,
        "speedup": 9999,
        "records": [
            {"reference_s": 2.0, "candidate_s": 1.0, "output_tokens": 100},
            {"reference_s": 4.0, "candidate_s": 2.0, "output_tokens": 100},
        ],
        **overrides,
    }


def test_benchmark_recomputes_score_and_keeps_correctness_a_producer_claim():
    result = paired_benchmark(artifact(), name="candidate", observed_ns=1)
    assert result["speedup"] == 2
    assert result["latency_reduction_fraction"] == 0.5
    assert result["reference_tokens_per_second"] == pytest.approx(200 / 6)
    assert result["candidate_tokens_per_second"] == pytest.approx(200 / 3)
    assert result["correctness"] == "reported_pass"
    failed = paired_benchmark(artifact(passed=False), name="failed", observed_ns=1)
    assert failed["correctness"] == "unverified"
    assert failed["speedup"] == 2


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "1"])
def test_invalid_timings_cannot_create_a_performance_claim(value):
    with pytest.raises(ValueError):
        paired_benchmark(
            artifact(records=[{"reference_s": 2, "candidate_s": value}]), name="bad", observed_ns=1
        )


def test_benchmark_missing_tokens_means_unknown_throughput():
    result = paired_benchmark(
        artifact(records=[{"reference_s": 2, "candidate_s": 1}]), name="test", observed_ns=1
    )
    assert result["candidate_tokens_per_second"] is None


def test_watch_reads_only_valid_benchmarks_and_skips_symlinks(tmp_path):
    (tmp_path / "candidate.json").write_text(json.dumps(artifact()))
    (tmp_path / "not-a-benchmark.json").write_text('{"secret": "not imported"}')
    (tmp_path / "bad.json").write_text(json.dumps(artifact(records=[])))
    (tmp_path / "linked.json").symlink_to(tmp_path / "candidate.json")
    runs, issues = read_benchmarks(tmp_path)
    assert len(runs) == 1
    assert runs[0]["name"] == "candidate"
    assert len(issues) == 1
    assert "secret" not in json.dumps(runs)


def test_demo_is_marked_synthetic_and_has_distinct_deployment_captures():
    state = Dashboard(demo=True).state()
    assert state["mode"] == "synthetic_demo"
    assert all(b["data_kind"] == "synthetic" for b in state["benchmarks"])
    assert all(r["metadata"]["data_kind"] == "synthetic" for r in state["reports"])
    a, b = state["reports"]
    assert a["gpus"][0]["estimated_engine_idle_seconds"] == pytest.approx(
        b["gpus"][0]["estimated_engine_idle_seconds"] * 2
    )


def test_readonly_dashboard_does_not_create_missing_databases(tmp_path):
    missing = tmp_path / "missing.sqlite"
    state = Dashboard(databases=[missing]).state()
    assert state["issues"] == [{"kind": "capture_unavailable", "name": "missing.sqlite"}]
    assert not missing.exists()


def test_dashboard_reads_ongoing_capture_and_discovers_new_benchmark(tmp_path):
    cfg = Config(
        "worker",
        "vllm",
        "http://localhost:8000/metrics",
        "http://localhost:9400/metrics",
        (GPU("gpu0", {"UUID": "GPU-DEMO"}),),
    )
    db = tmp_path / "capture.sqlite"
    with Store(db) as store:
        run = store.start(cfg)
        for seq, frame in enumerate(demo_frames(cfg)):
            store.append(run, seq, frame)
        dashboard = Dashboard(databases=[db], benchmark_dir=tmp_path)
        state = dashboard.state()
        assert state["reports"][0]["status"] == "collecting"
        assert state["reports"][0]["frames"] == 10
        assert not state["benchmarks"]
        (tmp_path / "result.json").write_text(json.dumps(artifact()))
        assert dashboard.state()["benchmarks"][0]["speedup"] == 2
        with Store(db, readonly=True) as readonly:
            with pytest.raises(Exception, match="readonly"):
                readonly.finish(run)


def test_demo_cannot_mix_with_real_telemetry():
    assert main(["dashboard", "--demo", "--monitor-local"]) == 2
    assert main(["dashboard", "--demo", "--db", "real.sqlite"]) == 2


def test_packaged_assets_exist_and_do_not_load_external_cdn():
    from importlib.resources import files

    web = files("obsrv").joinpath("web")
    html = web.joinpath("index.html").read_text()
    assert 'src="/app.js"' in html
    assert "https://" not in html
    assert "createElement" in web.joinpath("app.js").read_text()
    assert web.joinpath("style.css").is_file()


def test_apple_counter_is_system_wide_and_never_called_dcgm_idle():
    def runner(command, **options):
        assert command[0] == "/usr/sbin/ioreg"
        assert options["timeout"] == 3
        return SimpleNamespace(
            stdout=plistlib.dumps(
                [
                    {
                        "PerformanceStatistics": {
                            "Device Utilization %": 35,
                            "In use system memory": 1024,
                        }
                    }
                ]
            )
        )

    result = local_sample(runner=runner)
    # Also meaningful in Linux CI: Apple collection is not attempted there.
    if result["platform"] == "Darwin":
        assert result["gpu_activity_percent"] == 35
        assert result["gpu_memory_bytes"] == 1024
        assert result["status"] == "available"
    assert "system-wide" in result["scope"]
    assert "not DCGM" in result["measurement"]
    assert "idle_fraction" not in result


def test_apple_read_failure_remains_unknown():
    def failure(*args, **kwargs):
        raise OSError("not permitted")

    assert local_sample(runner=failure)["gpu_activity_percent"] is None
