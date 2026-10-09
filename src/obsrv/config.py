"""Explicit serving ownership and local artifact access; endpoints are never exported."""

import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .models import Identity


def endpoint(value: str):
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("endpoint must be an HTTP(S) URL")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("endpoint cannot contain credentials, query or fragment")


@dataclass(frozen=True)
class Config:
    identity: Identity
    serving_url: str
    trace_dir: Path
    state_dir: Path = Path("state")
    serving_labels: dict[str, str] = field(default_factory=dict)
    dcgm_url: str | None = None
    gpu_selectors: tuple[dict[str, str], ...] = ()
    trace_patterns: dict[str, str] = field(default_factory=dict)
    server_trace_dir: str | None = None
    auth_env: str | None = None
    interval_seconds: float = 5.0
    request_timeout_seconds: float = 5.0
    flush_timeout_seconds: float = 900.0
    retention_days: float = 7.0
    max_storage_bytes: int = 5 * 1024**3
    max_gap_seconds: float = 15.0

    def __post_init__(self):
        endpoint(self.serving_url)
        if self.dcgm_url:
            endpoint(self.dcgm_url)
            if not self.gpu_selectors:
                raise ValueError("DCGM requires explicit GPU UUID selectors")
        for selector in self.gpu_selectors:
            if not selector.get("UUID"):
                raise ValueError("GPU selectors require UUID; add GPU_I_ID for MIG")
        for i, first in enumerate(self.gpu_selectors):
            for second in self.gpu_selectors[i + 1 :]:
                if first["UUID"] == second["UUID"] and (
                    not first.get("GPU_I_ID")
                    or not second.get("GPU_I_ID")
                    or first["GPU_I_ID"] == second["GPU_I_ID"]
                ):
                    raise ValueError("GPU selectors overlap")
        for worker, pattern in self.trace_patterns.items():
            if (
                worker not in self.identity.workers
                or not pattern
                or Path(pattern).is_absolute()
                or ".." in Path(pattern).parts
            ):
                raise ValueError("trace_patterns require declared workers and safe relative globs")
        for selector in (self.serving_labels, *self.gpu_selectors):
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in selector.items()):
                raise ValueError("selectors must be string-to-string mappings")
        for name in (
            "interval_seconds",
            "request_timeout_seconds",
            "flush_timeout_seconds",
            "retention_days",
            "max_gap_seconds",
            "max_storage_bytes",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def load(cls, path: str | Path):
        data = (
            json.loads(Path(path).read_text())
            if Path(path).suffix == ".json"
            else tomllib.loads(Path(path).read_text())
        )
        raw = dict(data.pop("identity"))
        if not isinstance(raw["workers"], list):
            raise ValueError("identity.workers must be an array of worker names")
        raw["workers"] = tuple(raw["workers"])
        identity = Identity(**raw)
        for key in ("trace_dir", "state_dir"):
            if key in data:
                data[key] = Path(data[key])
        if "gpu_selectors" in data:
            data["gpu_selectors"] = tuple(data["gpu_selectors"])
        return cls(identity=identity, **data)
