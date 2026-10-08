"""Explicit endpoint and GPU ownership declarations. No automatic label joining."""

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def _keys(data, allowed, scope):
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {scope} keys: {sorted(unknown)}")


def _labels(value):
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("Label selectors must be string-to-string tables")
    return dict(value)


@dataclass(frozen=True)
class GPU:
    id: str
    labels: dict[str, str]


@dataclass(frozen=True)
class Config:
    deployment: str
    engine: str
    serving_url: str
    dcgm_url: str
    gpus: tuple[GPU, ...]
    serving_labels: dict[str, str] = field(default_factory=dict)
    allocation: str = "shared"
    interval_seconds: float = 1.0
    timeout_seconds: float = 2.0
    max_gap_seconds: float = 3.0
    join_tolerance_seconds: float = 0.5
    low_activity_threshold: float = 0.1
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.deployment, str) or not self.deployment:
            raise ValueError("deployment must be a nonempty string")
        if self.engine not in {"vllm", "sglang"}:
            raise ValueError("engine must be vllm or sglang")
        if self.allocation not in {"exclusive", "shared"}:
            raise ValueError("allocation must be exclusive or shared")
        for url in (self.serving_url, self.dcgm_url):
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("Metrics endpoints must be HTTP(S) URLs")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("URLs cannot contain credentials, query strings, or fragments")
        for name in (
            "interval_seconds",
            "timeout_seconds",
            "max_gap_seconds",
            "join_tolerance_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_gap_seconds < self.interval_seconds:
            raise ValueError("max_gap_seconds must be at least interval_seconds")
        if not 0 <= self.low_activity_threshold <= 1:
            raise ValueError("low_activity_threshold must be in [0, 1]")
        if not self.gpus or len({g.id for g in self.gpus}) != len(self.gpus):
            raise ValueError("Declare at least one GPU with unique ids")
        selectors = []
        for gpu in self.gpus:
            if not gpu.id or not _labels(gpu.labels).get("UUID"):
                raise ValueError("Each GPU selector requires a UUID (physical GPU or MIG entity)")
            selectors.append(tuple(sorted(gpu.labels.items())))
        if len(set(selectors)) != len(selectors):
            raise ValueError("GPU selectors must be distinct")
        for i, first in enumerate(self.gpus):
            for second in self.gpus[i + 1 :]:
                if first.labels["UUID"] == second.labels["UUID"] and (
                    not first.labels.get("GPU_I_ID")
                    or not second.labels.get("GPU_I_ID")
                    or first.labels["GPU_I_ID"] == second.labels["GPU_I_ID"]
                ):
                    raise ValueError("Cannot count a physical GPU and its MIG children together")
        _labels(self.serving_labels)
        _labels(self.metadata)

    def manifest(self):
        from dataclasses import asdict

        return asdict(self)

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    _keys(data, Config.__dataclass_fields__, "config")
    gpus = []
    for entry in data.pop("gpus", []):
        _keys(entry, {"id", "labels"}, "GPU")
        gpus.append(GPU(entry["id"], _labels(entry["labels"])))
    return Config(**data, gpus=tuple(gpus))
