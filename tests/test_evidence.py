import json
import time
from dataclasses import replace

import pytest

from obsrv import (
    Collector,
    EvidenceStore,
    StorageLimitError,
    build_bundle,
    export_context,
    import_trace,
    report,
    verify_export,
)
from obsrv.evidence import MAX_CONTEXT_CHARS, context_summary


def test_portable_export_integrity_retention_and_no_experiment_acceptance(store, config, trace):
    import_trace(store, config.identity, trace, profiler="torch", worker="rank0")
    path = export_context(store, config.identity)
    assert verify_export(path)
    data = json.loads((path / "bundle.json").read_text())
    assert data["version"] == 1
    assert data["source"] == "production_observations"
    assert data["official_gate_acceptance"] == "not_evaluated"
    assert data["observation_period"] == {"start": None, "end": None}
    assert "accepted_round" not in json.dumps(data)
    assert len((path / "context.md").read_text()) <= MAX_CONTEXT_CHARS
    store.prune(now=time.time() + 36501 * 86400)
    assert store.rows("profiles") == []
    assert verify_export(path)
    assert path.exists()
    artifact = path / data["artifacts"][0]["path"]
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact digest"):
        verify_export(path)


def test_large_context_truncates_explicitly_without_truncating_bundle(store, config, trace):
    profile = import_trace(store, config.identity, trace, profiler="torch", worker="rank0")
    bundle = build_bundle(store, config.identity)
    bundle.profiles[0]["reports"]["summary"] = "a" * 50000
    text = context_summary(bundle)
    assert len(text) == MAX_CONTEXT_CHARS
    assert "Context truncated" in text
    assert len(bundle.profiles[0]["reports"]["summary"]) == 50000
    assert profile["correctness"] == "not_evaluated"


def test_quota_interruptions_visible_and_exports_preserved(config, trace):
    store = EvidenceStore(config.state_dir, max_bytes=1)
    try:
        with pytest.raises(StorageLimitError):
            import_trace(store, config.identity, trace, profiler="torch", worker="rank0")
        bundle = build_bundle(store, config.identity)
        assert any("quota" in item for item in bundle.limitations)
        assert bundle.collection_events[0]["event_kind"] == "storage_limit"
    finally:
        store.close()


def test_version_reports_remain_separate(config, store):
    class Empty:
        def get(self, *args):
            return ""

    Collector(config, store, Empty()).collect_once()
    new = replace(config, identity=replace(config.identity, revision="optimized"))
    Collector(new, store, Empty()).collect_once()
    result = report(store)
    assert len(result["versions"]) == 2
    assert all(version["observation_count"] == 1 for version in result["versions"])
    assert "not paired" in result["comparison_policy"]
