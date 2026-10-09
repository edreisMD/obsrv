"""Passive collection; no profile is triggered by this collector."""

import threading
import time

from .config import Config
from .http import HttpTransport, Transport
from .metrics import GPU_NAMES, derive, gauge, parse, serving_names
from .models import Metric
from .store import EvidenceStore


class Collector:
    def __init__(self, config: Config, store: EvidenceStore, transport: Transport | None = None):
        self.config = config
        self.store = store
        self.transport = transport or HttpTransport(config.auth_env)
        self.gpu_transport = HttpTransport()
        self.previous = None
        self.previous_time = None
        self.previous_identity = None

    def reconfigure(self, config: Config):
        if config.serving_url != self.config.serving_url or config.auth_env != self.config.auth_env:
            self.previous = None
            self.transport = HttpTransport(config.auth_env)
        self.config = config

    def collect_once(self, *, now=None):
        config = self.config
        now = time.time() if now is None else now
        names = serving_names(config.identity.engine)
        coverage = []
        try:
            text = self.transport.get(
                config.serving_url.rstrip("/") + "/metrics", config.request_timeout_seconds
            )
            current = parse(
                text,
                {n for aliases in names.values() for n in aliases},
                now,
                config.max_gap_seconds,
            )
        except Exception as exc:
            current = []
            coverage.append("serving scrape failed: " + type(exc).__name__)
        start = self.previous_time if self.previous_time is not None else now
        elapsed = now - start
        reason = None
        if self.previous is None:
            reason = "first scrape; no interval baseline"
        elif self.previous_identity != config.identity.fingerprint:
            reason = "deployment identity changed; new interval baseline"
            start = now
        elif elapsed <= 0 or elapsed > config.max_gap_seconds:
            reason = "collection gap or non-monotonic timestamp"
        if reason:
            coverage.append(reason)
        metrics = derive(
            config.identity.engine,
            self.previous or [],
            current,
            config.serving_labels,
            elapsed,
            reason,
        )
        if config.dcgm_url:
            try:
                gpu = parse(
                    self.gpu_transport.get(config.dcgm_url, config.request_timeout_seconds),
                    {n for aliases, _ in GPU_NAMES.values() for n in aliases},
                    now,
                    config.max_gap_seconds,
                )
            except Exception as exc:
                gpu = []
                coverage.append("DCGM scrape failed: " + type(exc).__name__)
            for selector in config.gpu_selectors:
                for name, (aliases, unit) in GPU_NAMES.items():
                    value = gauge(gpu, aliases, selector)
                    metrics.append(
                        Metric(
                            name,
                            value,
                            unit,
                            "dcgm",
                            labels=selector,
                            unavailable_reason="missing/ambiguous/invalid GPU series"
                            if value is None
                            else None,
                        )
                    )
        record = {
            "start": start,
            "end": now,
            "source": "production_observation",
            "metrics": [m.record() for m in metrics],
            "coverage": coverage,
            "diagnostic": self.store.diagnostic_overlap(config.identity.deployment, start, now),
            "workload": {"source": "aggregate serving metrics", "request_content": "not_collected"},
        }
        self.store.observation(config.identity, record)
        self.previous, self.previous_time = current, now
        self.previous_identity = config.identity.fingerprint
        return record

    def run(self, stop: threading.Event | None = None, *, count=None):
        stop = stop or threading.Event()
        done = 0
        while not stop.is_set() and (count is None or done < count):
            self.collect_once()
            done += 1
            if count is None or done < count:
                stop.wait(self.config.interval_seconds)
