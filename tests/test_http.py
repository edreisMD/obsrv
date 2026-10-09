import gzip
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from obsrv import (
    Collector,
    EvidenceStore,
    ProfileController,
    activate,
    export_context,
    verify_export,
)
from obsrv.http import HttpTransport, ProfileRejectedError


@contextmanager
def server(trace, engine="vllm", *, compressed=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            if self.path == "/v1/models":
                self.respond(json.dumps({"data": [{"id": "test/model"}]}))
            elif self.path in {"/version", "/get_server_info"}:
                self.respond(json.dumps({"version": "fixture"}))
            elif self.path == "/metrics":
                self.server.scrapes += 1
                count = self.server.scrapes
                latency = f"{engine}:e2e_request_latency_seconds"
                self.respond(
                    f"# TYPE {engine}:generation_tokens counter\n"
                    f'{engine}:generation_tokens_total{{model_name="test/model"}} {100 * count}\n'
                    f"# TYPE {engine}:e2e_request_latency_seconds histogram\n"
                    f'{latency}_bucket{{model_name="test/model",le="1"}} {count}\n'
                    f'{latency}_bucket{{model_name="test/model",le="+Inf"}} {count}\n'
                    f'{latency}_count{{model_name="test/model"}} {count}\n'
                    f'{latency}_sum{{model_name="test/model"}} {0.5 * count}\n'
                )
            else:
                self.respond("{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.operations.append((self.path, body))
            if self.path == "/start_profile":
                self.server.body = body
            else:
                # Export finishing later than the scrape timeout is a distinct lifecycle phase.
                time.sleep(0.025)
                output = self.server.config.trace_dir
                if "output_dir" in self.server.body:
                    output = output / self.server.body["output_dir"].rsplit("/", 1)[-1]
                output.mkdir(parents=True, exist_ok=True)
                if compressed:
                    (output / "rank0.trace.json.gz").write_bytes(gzip.compress(trace.read_bytes()))
                else:
                    (output / "rank0.trace.json").write_bytes(trace.read_bytes())
            self.respond(json.dumps({"success": True}))

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    instance.operations = []
    instance.scrapes = 0
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join()


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_http_activation_collection_profile_and_portable_export(engine, tmp_path, trace):
    with server(trace, engine) as endpoint:
        url = f"http://127.0.0.1:{endpoint.server_port}"
        config, _, _ = activate(engine=engine, url=url, deployment="coding", state_dir=tmp_path)
        endpoint.config = config
        store = EvidenceStore(config.state_dir)
        try:
            collector = Collector(config, store)
            collector.collect_once()
            result = ProfileController(config, store).capture(duration_seconds=0.001)
            assert result["state"] == "partial"  # Topology discovery deliberately unverified.
            assert result["cleanup_confirmed"]
            assert result["profiles"]
            assert endpoint.operations[-1][0] == "/stop_profile"
            path = export_context(store, config.identity)
            assert verify_export(path)
        finally:
            store.close()


def test_success_false_body_is_not_an_accepted_profile_start(monkeypatch):
    transport = HttpTransport()
    monkeypatch.setattr(transport, "_request", lambda *args: '{"success": false}')
    with pytest.raises(ProfileRejectedError):
        transport.post("http://localhost/start_profile", {}, 1)
