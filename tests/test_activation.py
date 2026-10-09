import json
from pathlib import Path

import pytest

from obsrv import Config, activate, deployment_settings, verify_export
from obsrv.cli import main


class Metadata:
    def get(self, url, timeout):
        if url.endswith("/v1/models"):
            return json.dumps({"data": [{"id": "test/model"}]})
        return json.dumps(
            {
                "version": "fixture-version",
                "server_args": {"tp_size": 2, "api_key": "should-not-be-stored"},
            }
        )


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_easy_activation_and_safe_discovery(engine, tmp_path):
    config, path, guidance = activate(
        engine=engine,
        url="http://localhost:8000",
        deployment="coding",
        state_dir=tmp_path,
        transport=Metadata(),
    )
    assert Config.load(path) == config
    assert config.identity.model == "test/model"
    assert config.identity.workers == ("unattributed",)
    assert not config.identity.configuration["topology_verified"]
    assert "should-not-be-stored" not in path.read_text()
    assert config.identity.revision == "unknown"
    assert guidance["deployment_settings"]["engine"] == engine
    # Activation is idempotent and doesn't overwrite operator refinements.
    same, same_path, _ = activate(
        engine=engine,
        url="http://localhost:8000",
        deployment="coding",
        state_dir=tmp_path,
        transport=Metadata(),
    )
    assert same == config and same_path == path


def test_deployment_settings_enable_needed_shapes_without_launching(tmp_path):
    settings = deployment_settings("vllm", tmp_path)
    profiler = json.loads(settings["args"][1])
    assert profiler["torch_profiler_record_shapes"] is True
    assert profiler["torch_profiler_with_memory"] is True
    assert profiler["torch_profiler_dir"] == str(tmp_path)
    assert deployment_settings("sglang", tmp_path)["args"] == ["--enable-metrics"]


def test_activation_refuses_ambiguous_model_and_conflicting_identity(tmp_path):
    class Multiple:
        def get(self, *args):
            return json.dumps({"data": [{"id": "one"}, {"id": "two"}]})

    with pytest.raises(ValueError, match="ambiguous"):
        activate(engine="vllm", url="http://localhost", state_dir=tmp_path, transport=Multiple())
    activate(engine="vllm", url="http://localhost", state_dir=tmp_path, transport=Metadata())
    with pytest.raises(ValueError, match="different identity"):
        activate(
            engine="vllm",
            url="http://localhost",
            state_dir=tmp_path,
            revision="new",
            transport=Metadata(),
        )


def test_cli_import_export_report_and_verify(tmp_path, trace, capsys):
    config, path, _ = activate(
        engine="vllm", url="http://localhost", state_dir=tmp_path, transport=Metadata()
    )
    assert (
        main(
            [
                "import",
                "--config",
                str(path),
                str(trace),
                "--profiler",
                "torch",
                "--worker",
                "unattributed",
            ]
        )
        == 0
    )
    assert main(["report", "--config", str(path), "--all-versions"]) == 0
    capsys.readouterr()
    assert main(["export-context", "--config", str(path)]) == 0
    exported = Path(capsys.readouterr().out.strip())
    assert main(["verify", str(exported)]) == 0
    assert (exported / "bundle.json").exists()


def test_cli_activation_and_named_deployment_flow(tmp_path, trace, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("obsrv.activation.HttpTransport", lambda *args: Metadata())
    assert (
        main(
            [
                "activate",
                "--engine",
                "vllm",
                "--url",
                "http://localhost",
                "--deployment",
                "coding",
                "--setup-only",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "import",
                "--deployment",
                "coding",
                str(trace),
                "--profiler",
                "torch",
                "--worker",
                "unattributed",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["export-context", "--deployment", "coding"]) == 0
    assert verify_export(Path(capsys.readouterr().out.strip()))
