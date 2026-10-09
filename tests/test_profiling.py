import gzip
import json
import threading
import urllib.error
from dataclasses import replace

import pytest

from obsrv import ProfileController
from obsrv.http import ProfileRejectedError


class Engine:
    def __init__(self, config, trace, *, fail_stop=False):
        self.config, self.trace = config, trace
        self.fail_stop = fail_stop
        self.calls = []
        self.output = config.trace_dir

    def post(self, url, body, timeout):
        self.calls.append((url.rsplit("/", 1)[-1], body, timeout))
        if url.endswith("start_profile"):
            if "output_dir" in body:
                self.output = self.config.trace_dir / body["output_dir"].rsplit("/", 1)[-1]
        else:
            if self.fail_stop:
                raise TimeoutError("private server body")
            self.output.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.output / "rank0.trace.json.gz", "wt") as file:
                file.write(self.trace.read_text())


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_native_capture_worker_import_and_separate_flush_timeout(engine, config, store, trace):
    config = replace(config, identity=replace(config.identity, engine=engine))
    transport = Engine(config, trace)
    session = ProfileController(config, store, transport).capture(duration_seconds=0.001)
    assert session["state"] == "completed"
    assert session["cleanup_confirmed"]
    assert session["missing_workers"] == []
    assert len(session["profiles"]) == 1
    assert transport.calls[-1] == ("stop_profile", {}, config.flush_timeout_seconds)
    if engine == "sglang":
        assert transport.calls[0][1]["record_shapes"]
        assert transport.calls[0][1]["merge_profiles"] is False
    assert store.diagnostic_overlap(
        config.identity.deployment, session["created"], session["finished"]
    )


def test_cancel_stops_and_imports(config, store, trace):
    cancelled = threading.Event()
    cancelled.set()
    transport = Engine(config, trace)
    result = ProfileController(config, store, transport).capture(cancel=cancelled)
    assert result["state"] == "cancelled"
    assert result["cleanup_confirmed"]
    assert len(transport.calls) == 2


def test_partial_workers_and_ambiguous_mapping(config, store, trace):
    identity = replace(config.identity, workers=("rank0", "rank1"))
    config = replace(
        config, identity=identity, trace_patterns={"rank0": "rank0.*", "rank1": "rank1.*"}
    )
    result = ProfileController(config, store, Engine(config, trace)).capture(duration_seconds=0.001)
    assert result["state"] == "partial"
    assert result["missing_workers"] == ["rank1"]


def test_stop_timeout_survives_restart_and_blocks_new_capture(config, store, trace):
    broken = Engine(config, trace, fail_stop=True)
    first = ProfileController(config, store, broken).capture(duration_seconds=0.001)
    assert first["state"] == "cleanup_required"
    assert "private" not in json.dumps(first)
    controller = ProfileController(config, store, Engine(config, trace))
    with pytest.raises(RuntimeError, match="recovery"):
        controller.capture(duration_seconds=0.001)
    result = controller.recover(first["id"])
    assert result["state"] == "recovered"
    assert store.outstanding(config.identity.deployment) == []


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("http://localhost", 400, "already running", {}, None),
        ProfileRejectedError("already running"),
    ],
)
def test_rejected_start_never_stops_foreign_capture(config, store, error):
    class Rejected:
        def __init__(self):
            self.calls = []

        def post(self, url, body, timeout):
            self.calls.append(url)
            raise error

    transport = Rejected()
    result = ProfileController(config, store, transport).capture(duration_seconds=0.001)
    assert result["state"] == "failed"
    assert result["cleanup_confirmed"]
    assert len(transport.calls) == 1


def test_interrupt_cleanup_and_lock_ownership(config, store, trace):
    class Interrupt(Engine):
        def post(self, url, body, timeout):
            if url.endswith("start_profile"):
                raise KeyboardInterrupt
            super().post(url, body, timeout)

    with pytest.raises(KeyboardInterrupt):
        ProfileController(config, store, Interrupt(config, trace)).capture()
    assert not store.outstanding(config.identity.deployment)
    with store.capture_lock(config.identity.deployment):
        with pytest.raises(RuntimeError, match="owns"):
            ProfileController(config, store, Engine(config, trace)).capture()


def test_changed_identity_cannot_recover_another_revision(config, store, trace):
    first = ProfileController(config, store, Engine(config, trace, fail_stop=True)).capture(
        duration_seconds=0.001
    )
    controller = ProfileController(
        replace(config, identity=replace(config.identity, revision="new")),
        store,
        Engine(config, trace),
    )
    with pytest.raises(ValueError, match="original"):
        controller.recover(first["id"])
