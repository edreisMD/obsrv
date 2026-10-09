"""JSON-safe production contracts. No experimental acceptance is implied."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Identity:
    deployment: str
    revision: str
    engine: str
    engine_version: str
    model: str
    configuration: dict[str, Any]
    hardware: dict[str, Any]
    workers: tuple[str, ...]

    def __post_init__(self):
        if self.engine not in {"vllm", "sglang"}:
            raise ValueError("engine must be vllm or sglang")
        for name in ("deployment", "revision", "engine_version", "model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"identity.{name} must be nonempty")
        if not isinstance(self.configuration, dict) or not isinstance(self.hardware, dict):
            raise ValueError("identity configuration and hardware must be mappings")
        if not isinstance(self.workers, (tuple, list)):
            raise ValueError("identity.workers must be a sequence of worker names")
        if not self.workers or len(set(self.workers)) != len(self.workers):
            raise ValueError("identity.workers must be nonempty and unique")
        if not all(isinstance(w, str) and w for w in self.workers):
            raise ValueError("identity.workers must contain nonempty strings")
        # Exportable deployment configuration must be a sanitized, operator-provided description.
        _reject_secrets(asdict(self))
        canonical(asdict(self))

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    def record(self) -> dict:
        return asdict(self)


def _reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(
                s in key.lower() for s in ("password", "secret", "api_key", "authorization")
            ) or key.lower() in {"token", "access_token", "hf_token", "credentials", "apikey"}:
                raise ValueError(f"secret-bearing identity field is not exportable: {key}")
            _reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item)


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | None
    unit: str
    source: str
    direction: str | None = None
    unavailable_reason: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric value must be finite or unavailable")
        if self.direction not in {None, "min", "max"}:
            raise ValueError("metric direction must be min, max or null")
        if self.value is None and not self.unavailable_reason:
            raise ValueError("unavailable metric needs a reason")

    def record(self):
        return asdict(self)
