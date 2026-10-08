"""Supplemental serving metrics; the primary measurement source is the VibeSys profiler."""

import math
from dataclasses import dataclass

from prometheus_client.parser import text_string_to_metric_families

from .models import Metric


@dataclass(frozen=True)
class Sample:
    name: str
    labels: dict[str, str]
    value: float | None


def serving_names(engine):
    prefix = engine + ":"
    raw = {
        "running": ("num_requests_running",) if engine == "vllm" else ("num_running_reqs",),
        "waiting": ("num_requests_waiting",) if engine == "vllm" else ("num_queue_reqs",),
        "kv_cache_fraction": ("kv_cache_usage_perc", "gpu_cache_usage_perc")
        if engine == "vllm"
        else ("token_usage",),
        "generation_tokens": ("generation_tokens_total",),
        "prompt_tokens": ("prompt_tokens_total",),
        "completed_requests": ("request_success_total", "e2e_request_latency_seconds_count")
        if engine == "vllm"
        else ("e2e_request_latency_seconds_count",),
        "errors": ("request_failure_total",),
    }
    for field, stem in {
        "ttft": "time_to_first_token_seconds",
        "tpot": "time_per_output_token_seconds",
        "itl": "inter_token_latency_seconds",
        "e2e": "e2e_request_latency_seconds",
        "queue": "request_queue_time_seconds" if engine == "vllm" else "queue_time_seconds",
        "input_length": "request_prompt_tokens" if engine == "vllm" else "prompt_tokens",
        "output_length": "request_generation_tokens" if engine == "vllm" else "generation_tokens",
    }.items():
        for suffix in ("bucket", "sum", "count"):
            raw[f"{field}_{suffix}"] = (stem + "_" + suffix,)
    return {key: tuple(prefix + name for name in aliases) for key, aliases in raw.items()}


GPU_NAMES = {
    "gpu_utilization": (("DCGM_FI_DEV_GPU_UTIL",), "percent"),
    "gpu_memory_used": (("DCGM_FI_DEV_FB_USED",), "MiB"),
    "gpu_power": (("DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_BOARD_POWER_WATTS"), "W"),
    "gpu_sm_active": (("DCGM_FI_PROF_SM_ACTIVE",), "fraction"),
    "gpu_dram_active": (("DCGM_FI_PROF_DRAM_ACTIVE",), "fraction"),
    "gpu_tensor_active": (("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",), "fraction"),
}


def parse(text, wanted, now, max_age):
    result = []
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name not in wanted:
                continue
            value = float(s.value)
            if (
                not math.isfinite(value)
                or value < 0
                or (s.name.startswith("DCGM_") and value >= 9e15)
            ):
                value = None
            if s.timestamp is not None and abs(now - float(s.timestamp)) > max_age:
                value = None
            result.append(Sample(s.name, dict(s.labels), value))
    return result


def select(samples, aliases, selector):
    for name in aliases:
        result = [
            s
            for s in samples
            if s.name == name and all(s.labels.get(k) == v for k, v in selector.items())
        ]
        if result:
            return result
    return []


def _index(samples, aliases, selector):
    selected = select(samples, aliases, selector)
    keys = [(s.name, tuple(sorted(s.labels.items()))) for s in selected]
    if len(keys) != len(set(keys)):
        return {}
    return {key: s.value for key, s in zip(keys, selected, strict=True)}


def delta(before, after, aliases, selector):
    old, new = _index(before, aliases, selector), _index(after, aliases, selector)
    if not old or old.keys() != new.keys():
        return None
    if any(old[k] is None or new[k] is None or new[k] < old[k] for k in old):
        return None
    return sum(new[k] - old[k] for k in old)


def gauge(samples, aliases, selector):
    selected = select(samples, aliases, selector)
    if len(selected) != 1:
        return None
    return selected[0].value


def histogram(before, after, aliases, selector, quantile):
    """Classic histogram quantile of interval deltas, not lifetime buckets."""
    old, new = _index(before, aliases, selector), _index(after, aliases, selector)
    if not old or old.keys() != new.keys():
        return None
    buckets = {}
    bounds_by_series = {}
    for key in old:
        if old[key] is None or new[key] is None or new[key] < old[key]:
            return None
        try:
            bound = float(dict(key[1])["le"])
        except (KeyError, ValueError):
            return None
        if math.isnan(bound) or bound < 0:
            return None
        labels = tuple((k, v) for k, v in key[1] if k != "le")
        bounds_by_series.setdefault(labels, set()).add(bound)
        buckets[bound] = buckets.get(bound, 0) + new[key] - old[key]
    if any(bounds != set(buckets) for bounds in bounds_by_series.values()):
        return None  # Different worker schemas cannot be summed into one histogram.
    if math.inf not in buckets or buckets[math.inf] <= 0:
        return None
    ordered = sorted(buckets.items())
    if any(b < a for (_, a), (_, b) in zip(ordered, ordered[1:], strict=False)):
        return None
    target = quantile * buckets[math.inf]
    low, previous = 0.0, 0.0
    for bound, count in ordered:
        if count >= target:
            if math.isinf(bound):
                return None  # Tail has no finite bound; don't invent a finite p95.
            return low + (bound - low) * (target - previous) / (count - previous)
        low, previous = bound, count
    return None


def derive(engine, before, after, selector, elapsed, interval_reason=None):
    names = serving_names(engine)
    result = []
    reason = "missing, ambiguous, stale or invalid series"
    for key in ("running", "waiting", "kv_cache_fraction"):
        value = gauge(after, names[key], selector)
        if key == "kv_cache_fraction" and value is not None and value > 1:
            value = None
        result.append(
            Metric(
                key,
                value,
                "fraction" if key.endswith("fraction") else "requests",
                "serving",
                unavailable_reason=reason if value is None else None,
            )
        )
    for key in ("completed_requests", "generation_tokens", "prompt_tokens", "errors"):
        value = None if interval_reason else delta(before, after, names[key], selector)
        if value is not None:
            value /= elapsed
        unit = "tokens/s" if "tokens" in key else "requests/s"
        result.append(
            Metric(
                key + "_rate",
                value,
                unit,
                "serving",
                unavailable_reason=(interval_reason or "missing/reset/changed series")
                if value is None
                else None,
            )
        )
    for key in ("ttft", "tpot", "itl", "e2e", "queue", "input_length", "output_length"):
        unit = "tokens" if key.endswith("length") else "s"
        count = None if interval_reason else delta(before, after, names[key + "_count"], selector)
        total = None if interval_reason else delta(before, after, names[key + "_sum"], selector)
        mean = total / count if total is not None and count else None
        result.append(
            Metric(
                key + "_mean",
                mean,
                unit,
                "serving",
                unavailable_reason=(interval_reason or "no valid interval observations")
                if mean is None
                else None,
            )
        )
        for q, label in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
            value = (
                None
                if interval_reason
                else histogram(before, after, names[key + "_bucket"], selector, q)
            )
            result.append(
                Metric(
                    key + "_" + label,
                    value,
                    unit,
                    "serving",
                    "min" if unit == "s" else None,
                    (interval_reason or "no valid finite interval histogram")
                    if value is None
                    else None,
                )
            )
    return result
