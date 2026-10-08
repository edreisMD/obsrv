from dataclasses import replace
from pathlib import Path

import pytest

from obsrv import Collector
from obsrv.metrics import Sample, delta, histogram


def buckets(a, b, c):
    return [
        Sample("x_bucket", {"le": bound}, value)
        for bound, value in zip(("1", "2", "+Inf"), (a, b, c), strict=True)
    ]


def test_histogram_uses_interval_not_lifetime():
    assert histogram(
        buckets(100, 100, 100), buckets(100, 110, 110), ("x_bucket",), {}, 0.95
    ) == pytest.approx(1.95)
    assert histogram(buckets(0, 0, 0), buckets(0, 0, 10), ("x_bucket",), {}, 0.95) is None


@pytest.mark.parametrize("after", [buckets(0, 1, 2), buckets(100, 105, 104), []])
def test_reset_changed_and_invalid_histogram_unavailable(after):
    assert histogram(buckets(100, 100, 100), after, ("x_bucket",), {}, 0.95) is None


def test_duplicate_and_reset_counters_unavailable():
    before = [Sample("x", {}, 10)]
    assert delta(before, [Sample("x", {}, 2)], ("x",), {}) is None
    assert delta(before, [Sample("x", {}, 11), Sample("x", {}, 11)], ("x",), {}) is None


class Scrapes:
    def __init__(self, text):
        self.text = text

    def get(self, url, timeout):
        return self.text


def metric(record, name):
    return next(m for m in record["metrics"] if m["name"] == name)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_engine_scrapes_rollout_gaps_missing_and_rates(engine, config, store):
    config = replace(config, identity=replace(config.identity, engine=engine))
    text = (Path(__file__).parent / f"fixtures/{engine}.prom").read_text()
    transport = Scrapes(text)
    collector = Collector(config, store, transport)
    first = collector.collect_once(now=100)
    assert metric(first, "generation_tokens_rate")["value"] is None
    transport.text = text.replace(" 100\n", " 110\n").replace(" 200\n", " 220\n")
    second = collector.collect_once(now=105)
    assert metric(second, "generation_tokens_rate")["value"] == 2
    assert metric(second, "errors_rate")["value"] is None
    collector.reconfigure(replace(config, identity=replace(config.identity, revision="new")))
    rollout = collector.collect_once(now=110)
    assert rollout["start"] == rollout["end"]
    assert metric(rollout, "generation_tokens_rate")["value"] is None
    assert len(store.rows("identities")) == 2
    gap = collector.collect_once(now=200)
    assert metric(gap, "generation_tokens_rate")["value"] is None
    transport.text = ""
    assert metric(collector.collect_once(now=205), "running")["value"] is None


def test_scrape_failures_do_not_persist_payloads(config, store):
    class Broken:
        def get(self, *args):
            raise RuntimeError("secret prompt and authorization token")

    record = Collector(config, store, Broken()).collect_once()
    assert "secret" not in str(record)
    assert record["coverage"] == [
        "serving scrape failed: RuntimeError",
        "first scrape; no interval baseline",
    ]


def test_incompatible_worker_histograms_not_aggregated():
    old = [Sample("x_bucket", {"worker": "a", "le": b}, 0) for b in ("1", "+Inf")]
    old += [Sample("x_bucket", {"worker": "b", "le": b}, 0) for b in ("2", "+Inf")]
    new = [Sample(sample.name, sample.labels, 10) for sample in old]
    assert histogram(old, new, ("x_bucket",), {}, 0.5) is None
