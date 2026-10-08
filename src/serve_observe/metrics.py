"""Allowlisted metric adapters; ambiguous series and invalid values stay unknown."""

import math
from dataclasses import asdict, dataclass

from prometheus_client.parser import text_string_to_metric_families

GPU_METRICS = {
    "engine_active": ("DCGM_FI_PROF_GR_ENGINE_ACTIVE", "DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO"),
    "sm_active": ("DCGM_FI_PROF_SM_ACTIVE",),
    "dram_active": ("DCGM_FI_PROF_DRAM_ACTIVE",),
    "tensor_active": ("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",),
    "gpu_util_percent": ("DCGM_FI_DEV_GPU_UTIL",),
    "power_watts": ("DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_BOARD_POWER_WATTS"),
}


def serving_metrics(engine):
    if engine == "vllm":
        running, waiting = "num_requests_running", "num_requests_waiting"
        kv = ("kv_cache_usage_perc", "gpu_cache_usage_perc")
        queue = "request_queue_time_seconds"
    else:
        running, waiting, kv = "num_running_reqs", "num_queue_reqs", ("token_usage",)
        queue = "queue_time_seconds"
    prefix = engine + ":"
    names = {
        "running": (prefix + running,),
        "waiting": (prefix + waiting,),
        "kv_cache_fraction": tuple(prefix + x for x in kv),
        "generation_tokens": (prefix + "generation_tokens_total",),
        "prompt_tokens": (prefix + "prompt_tokens_total",),
    }
    for field, stem in {
        "ttft": "time_to_first_token_seconds",
        "itl": "inter_token_latency_seconds",
        "e2e": "e2e_request_latency_seconds",
        "queue": queue,
    }.items():
        for suffix in ("bucket", "sum", "count"):
            names[f"{field}_{suffix}"] = (prefix + stem + "_" + suffix,)
    return names


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float | None
    source_timestamp: float | None = None

    def record(self):
        return asdict(self)


def parse_metrics(text, names, now_seconds, max_age_seconds):
    wanted = {name for variants in names.values() for name in variants}
    result = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in wanted:
                continue
            value = float(sample.value)
            timestamp = float(sample.timestamp) if sample.timestamp is not None else None
            if not math.isfinite(value) or value < 0:
                value = None
            if value is not None and sample.name.startswith("DCGM_") and value >= 9e15:
                value = None  # DCGM blank/error sentinel encodings.
            if timestamp is not None and (
                not math.isfinite(timestamp)
                or now_seconds - timestamp > max_age_seconds
                or timestamp - now_seconds > max_age_seconds
            ):
                value = None
            result.append(Sample(sample.name, dict(sample.labels), value, timestamp))
    return result


def select(samples, names, labels):
    # Alias priority is explicit. Never sum aliases representing the same metric.
    for name in names:
        matching = [
            s
            for s in samples
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items())
        ]
        if matching:
            return matching
    return []


def gauge(samples, names, labels, *, ratio=False):
    selected = select(samples, names, labels)
    if len(selected) != 1 or selected[0].value is None:
        return None
    value = selected[0].value
    if ratio and value > 1:
        return None
    return value


def counter_delta(before, after, names, labels):
    def index(samples):
        selected = select(samples, names, labels)
        keys = [(s.name, tuple(sorted(s.labels.items()))) for s in selected]
        if len(set(keys)) != len(keys):
            return {}
        return {key: sample.value for key, sample in zip(keys, selected, strict=True)}

    old, new = index(before), index(after)
    if not old or old.keys() != new.keys():
        return None
    if any(old[k] is None or new[k] is None or new[k] < old[k] for k in old):
        return None  # Reset, missing series, or invalid metric: no fabricated work.
    return sum(new[k] - old[k] for k in old)
