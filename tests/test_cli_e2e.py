"""Real HTTP + separate CLI processes. Optionally exercise an isolated installed wheel."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_http import server


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_cli_handoff(engine, tmp_path, trace):
    executable = os.environ.get("OBSRV_E2E_CLI")
    command = [executable] if executable else [sys.executable, "-m", "obsrv"]
    env = dict(os.environ)
    if executable:
        # Installed CLI must work without the checkout or its editable installation.
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    def run(*args, status=0):
        result = subprocess.run(
            [*command, *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == status, result.stderr + result.stdout
        return (result.stdout + (result.stderr if status == 1 else "")).strip()

    run("--help")
    settings = json.loads(
        run("deployment-settings", "--engine", engine, "--trace-dir", str(tmp_path / "traces"))
    )
    assert settings
    with server(trace, engine, compressed=True) as endpoint:
        activation = json.loads(
            run(
                "activate",
                "--engine",
                engine,
                "--url",
                f"http://127.0.0.1:{endpoint.server_port}",
                "--deployment",
                "coding",
                "--revision",
                "sha256:fixture",
                "--trace-dir",
                str(tmp_path / "traces"),
                "--setup-only",
            )
        )
        config_path = Path(activation["config"])
        from obsrv import Config

        endpoint.config = Config.load(config_path)
        config = json.loads(config_path.read_text())
        config["interval_seconds"] = 0.01
        config_path.write_text(json.dumps(config))
        selection = ("--deployment", "coding")
        observations = [
            json.loads(line) for line in run("collect", *selection, "--count", "2").splitlines()
        ]
        values = {m["name"]: m["value"] for m in observations[-1]["metrics"]}
        assert values["generation_tokens_rate"] > 0
        assert values["e2e_mean"] == 0.5
        assert values["e2e_p95"] == pytest.approx(0.95)
        assert values["errors_rate"] is None

        # Auto-discovery must not silently certify unknown distributed worker coverage.
        capture = json.loads(run("profile", *selection, "--duration", "0.01", status=2))
        assert capture["state"] == "partial" and capture["cleanup_confirmed"]
        assert capture["profiles"]
        if engine == "sglang":
            assert endpoint.operations[0][1]["record_shapes"] is True
            assert endpoint.operations[0][1]["activities"] == ["CPU", "GPU"]
        else:
            assert endpoint.operations[0][1] == {}

        # Operator-declared rank mapping permits a fully covered single-worker test.
        config["identity"]["workers"] = ["rank0"]
        config["identity"]["configuration"]["topology_verified"] = True
        config["trace_patterns"] = {"rank0": "**/rank0.trace.json.gz"}
        config_path.write_text(json.dumps(config))
        endpoint.config = Config.load(config_path)
        capture = json.loads(run("profile", *selection, "--duration", "0.01"))
        assert capture["state"] == "completed" and not capture["missing_workers"]
        report = json.loads(run("report", *selection))
        assert report["versions"][0]["profiles"][0]["summary"]["total_cuda_time_us"] == 120
        exported = Path(run("export-context", *selection))
        # Move it away from SQLite and trace storage: a recipient needs only the bundle.
        portable = tmp_path / "recipient"
        shutil.copytree(exported, portable)
        assert run("verify", str(portable)) == "Export integrity verified"
        bundle = json.loads((portable / "bundle.json").read_text())
        assert bundle["source"] == "production_observations"
        assert bundle["official_gate_acceptance"] == "not_evaluated"
        assert len((portable / "context.md").read_text()) <= 16_384
        assert bundle["profiles"][0]["provenance"]
        assert bundle["artifacts"][0]["format"] == "chrome_trace_json_gzip"
        assert "accepted_round" not in bundle

        # VibeSys accepts raw traces, not ProductionEvidenceBundle. Restore the
        # compression suffix (its upstream reader uses the extension), then prove
        # the unchanged upstream analyzer can consume the exported bytes.
        artifact = bundle["artifacts"][0]
        upstream_trace = tmp_path / "handoff.trace.json.gz"
        shutil.copyfile(portable / artifact["path"], upstream_trace)
        reference = os.environ.get("VIBESYS_REFERENCE_ROOT")
        if reference:
            source = Path(reference) / "resources/profilers/torch/analyze_torch_profile.py"
            spec = importlib.util.spec_from_file_location("handoff_reference_" + engine, source)
            upstream = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = upstream
            spec.loader.exec_module(upstream)
            raw = upstream._read_json_maybe_gz(str(upstream_trace))
            assert upstream._summarize_chrome_trace(raw) == bundle["profiles"][0]["summary"]
        # Manual trace ingestion into obsrv also remains available.
        imported = json.loads(
            run(
                "import",
                *selection,
                str(upstream_trace),
                "--profiler",
                "torch",
                "--worker",
                "rank0",
            )
        )
        assert imported["summary"]["total_cuda_time_us"] == 120
        (portable / "context.md").write_text("tampered")
        assert "digest mismatch" in run("verify", str(portable), status=1)
