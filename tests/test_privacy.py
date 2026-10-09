import os
from dataclasses import replace

from obsrv import Collector, activate
from obsrv.http import HttpTransport


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.body


class Opener:
    def __init__(self):
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return Response(b"")


def test_serving_token_not_forwarded_to_dcgm(config, store, monkeypatch):
    monkeypatch.setenv("FIXTURE_SERVING_TOKEN", "synthetic-auth-value")
    config = replace(
        config,
        auth_env="FIXTURE_SERVING_TOKEN",
        dcgm_url="http://localhost:9400/metrics",
        gpu_selectors=({"UUID": "fixture"},),
    )
    collector = Collector(config, store)
    serving = Opener()
    gpu = Opener()
    collector.transport.opener = serving
    collector.gpu_transport.opener = gpu
    collector.collect_once()
    assert serving.requests[0].get_header("Authorization") == "Bearer synthetic-auth-value"
    assert gpu.requests[0].get_header("Authorization") is None
    assert "synthetic-auth-value" not in str(store.rows("observations"))


def test_activation_creates_private_config_before_writing(tmp_path, monkeypatch):
    class Metadata:
        def get(self, url, timeout):
            if url.endswith("/v1/models"):
                return '{"data":[{"id":"fixture-model"}]}'
            return '{"version":"fixture"}'

    real_open = os.open
    modes = []

    def inspect(path, flags, mode=0o777):
        modes.append(mode)
        return real_open(path, flags, mode)

    monkeypatch.setattr("obsrv.activation.os.open", inspect)
    _, path, _ = activate(
        engine="vllm", url="http://localhost", state_dir=tmp_path, transport=Metadata()
    )
    assert modes[-1] == 0o600
    assert path.stat().st_mode & 0o777 == 0o600


def test_http_redirect_does_not_forward_credentials():
    transport = HttpTransport("FIXTURE_SERVING_TOKEN")
    redirect = next(
        handler for handler in transport.opener.handlers if type(handler).__name__ == "_NoRedirect"
    )
    assert redirect.redirect_request(None, None, 302, "redirect", {}, "http://other") is None
