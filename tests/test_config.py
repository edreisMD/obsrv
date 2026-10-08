from dataclasses import replace

import pytest

from obsrv import Config


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/foo",
        "http://" + "user:password@" + "localhost",
        "http://localhost?token=value",
        "http://localhost/#fragment",
    ],
)
def test_credential_and_non_http_urls_rejected(config, url):
    with pytest.raises(ValueError):
        replace(config, serving_url=url)


def test_explicit_gpu_ownership_required(config):
    with pytest.raises(ValueError, match="explicit"):
        replace(config, dcgm_url="http://localhost:9400/metrics")
    with pytest.raises(ValueError, match="overlap"):
        replace(
            config,
            dcgm_url="http://localhost:9400/metrics",
            gpu_selectors=({"UUID": "gpu"}, {"UUID": "gpu", "GPU_I_ID": "1"}),
        )


def test_unknown_config_and_secret_identity_fields_rejected(config, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        (__import__("pathlib").Path(__file__).parents[1] / "examples/deployment.toml").read_text()
        + '\nunexpected = "bad"\n'
    )
    # This appends into hardware metadata, which is user-described; top-level keys are strict.
    path.write_text('unexpected = "bad"\n' + path.read_text())
    with pytest.raises(TypeError):
        Config.load(path)
    with pytest.raises(ValueError, match="secret-bearing"):
        replace(config.identity, configuration={"api_key": "private"})


def test_worker_string_and_credentials_rejected(config):
    with pytest.raises(ValueError, match="sequence"):
        replace(config.identity, workers="rank0")
    with pytest.raises(ValueError, match="secret-bearing"):
        replace(config.identity, configuration={"hf_token": "private"})
