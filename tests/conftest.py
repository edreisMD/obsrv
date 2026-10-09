import json
from pathlib import Path

import pytest

from obsrv import Config, EvidenceStore, Identity


@pytest.fixture
def identity():
    return Identity(
        "coding",
        "sha256:baseline",
        "vllm",
        "fixture-version",
        "test/model",
        {"dtype": "bfloat16", "tensor_parallel_size": 1},
        {"gpu": "H100", "count": 1},
        ("rank0",),
    )


@pytest.fixture
def config(tmp_path, identity):
    return Config(identity, "http://127.0.0.1:8000", tmp_path / "traces", tmp_path / "state")


@pytest.fixture
def store(config):
    result = EvidenceStore(config.state_dir, retention_days=36500)
    yield result
    result.close()


@pytest.fixture
def trace():
    return Path(__file__).parent / "fixtures/torch.json"


@pytest.fixture
def raw_trace(trace):
    return json.loads(trace.read_text())
