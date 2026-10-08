"""Time-weighted evidence. Estimates and correlated demand are not causal attribution."""

from itertools import pairwise

from .metrics import GPU_METRICS, counter_delta, gauge, serving_metrics


def _midpoint(scrape):
    # Keep twice the midpoint as integer nanoseconds until subtracting timestamps.
    return scrape.started_ns + scrape.ended_ns


def _interval(old, new, config):
    dt = (_midpoint(new) - _midpoint(old)) / 2e9
    if old.error or new.error or not 0 < dt <= config.max_gap_seconds:
        return None
    if any((s.ended_ns - s.started_ns) / 1e9 > config.max_gap_seconds for s in (old, new)):
        return None
    return dt


def _demand(frame, config, metrics):
    gpu, serving = frame["gpu"], frame["serving"]
    if gpu.error or serving.error:
        return None
    # Both acquisition envelopes must fit within the join tolerance.
    span = max(gpu.ended_ns, serving.ended_ns) - min(gpu.started_ns, serving.started_ns)
    if span / 1e9 > config.join_tolerance_seconds:
        return None
    running = gauge(serving.samples, metrics["running"], config.serving_labels)
    waiting = gauge(serving.samples, metrics["waiting"], config.serving_labels)
    return None if running is None or waiting is None else running + waiting


def analyze(config, frames):
    """Summarize a single worker's explicitly mapped GPUs without bridging outages.

    A current DCGM value represents the preceding interval in this estimate.
    Its real hardware watch window is exporter-controlled and may differ.
    """
    sources = []
    for frame in frames:
        by_source = {s.source: s for s in frame}
        if len(frame) != 2 or set(by_source) != {"gpu", "serving"}:
            raise ValueError("Each frame requires exactly one gpu and one serving scrape")
        sources.append(by_source)
    metrics = serving_metrics(config.engine)
    gpu_rows = []
    timeline = []
    for gpu in config.gpus:
        elapsed = covered = idle = no_demand = with_demand = unknown_demand = 0.0
        low_demand = demand_covered = 0.0
        sm_integral = sm_seconds = 0.0
        for old, new in pairwise(sources):
            dt = (_midpoint(new["gpu"]) - _midpoint(old["gpu"])) / 2e9
            if dt <= 0:
                raise ValueError("GPU scrape timestamps must increase within a run")
            elapsed += dt
            active = gauge(new["gpu"].samples, GPU_METRICS["engine_active"], gpu.labels, ratio=True)
            previous = gauge(
                old["gpu"].samples, GPU_METRICS["engine_active"], gpu.labels, ratio=True
            )
            valid = _interval(old["gpu"], new["gpu"], config) is not None
            if not valid or active is None or previous is None:
                continue
            covered += dt
            estimated_idle = dt * (1 - active)
            idle += estimated_idle
            demand = _demand(new, config, metrics)
            previous_demand = _demand(old, config, metrics)
            # Never call transitional demand intervals stable no-demand/with-demand.
            if demand is None or previous_demand is None or (demand > 0) != (previous_demand > 0):
                state = "unknown_or_transitioning_demand"
                unknown_demand += estimated_idle
            elif demand == 0:
                state = "no_demand"
                no_demand += estimated_idle
                demand_covered += dt
            else:
                state = "with_demand"
                with_demand += estimated_idle
                demand_covered += dt
                if active < config.low_activity_threshold:
                    low_demand += dt
            sm = gauge(new["gpu"].samples, GPU_METRICS["sm_active"], gpu.labels, ratio=True)
            if sm is not None:
                sm_integral += sm * dt
                sm_seconds += dt
            timeline.append(
                {
                    "gpu": gpu.id,
                    "wall_time_ns": new["gpu"].wall_time_ns,
                    "interval_seconds": dt,
                    "engine_active_fraction": active,
                    "estimated_engine_idle_seconds": estimated_idle,
                    "demand_state": state,
                    "requests_at_interval_end": demand,
                }
            )
        gpu_rows.append(
            {
                "gpu": gpu.id,
                "window_seconds": elapsed,
                "covered_seconds": covered,
                "coverage_fraction": covered / elapsed if elapsed else 0.0,
                "estimated_engine_idle_seconds": idle if covered else None,
                "estimated_engine_idle_fraction": idle / covered if covered else None,
                "idle_seconds_no_demand": no_demand if demand_covered else None,
                "idle_seconds_with_demand": with_demand if demand_covered else None,
                "idle_seconds_unknown_demand": unknown_demand if covered else None,
                "demand_classified_seconds": demand_covered,
                "low_activity_with_demand_seconds": low_demand if demand_covered else None,
                "mean_sm_active_fraction": sm_integral / sm_seconds if sm_seconds else None,
            }
        )

    serving = {"counter_intervals_skipped": 0}
    for field in (
        "generation_tokens",
        "prompt_tokens",
        "ttft_sum",
        "ttft_count",
        "itl_sum",
        "itl_count",
        "e2e_sum",
        "e2e_count",
        "queue_sum",
        "queue_count",
    ):
        total = covered = 0.0
        for old, new in pairwise(sources):
            dt = _interval(old["serving"], new["serving"], config)
            delta = counter_delta(
                old["serving"].samples,
                new["serving"].samples,
                metrics[field],
                config.serving_labels,
            )
            if dt is None or delta is None:
                if field == "generation_tokens":
                    serving["counter_intervals_skipped"] += 1
                continue
            total += delta
            covered += dt
        serving[field] = {
            "delta": total if covered else None,
            "covered_seconds": covered,
            "rate_per_second": total / covered if covered else None,
        }

    # Pair histogram sum/count deltas in the SAME intervals; no averaging of lifetime means.
    for field in ("ttft", "itl", "e2e", "queue"):
        total = count = covered = 0.0
        for old, new in pairwise(sources):
            dt = _interval(old["serving"], new["serving"], config)
            sums, counts = (
                counter_delta(
                    old["serving"].samples,
                    new["serving"].samples,
                    metrics[field + suffix],
                    config.serving_labels,
                )
                for suffix in ("_sum", "_count")
            )
            if dt is not None and sums is not None and counts is not None:
                total += sums
                count += counts
                covered += dt
        serving[field] = {
            "mean_seconds": total / count if count else None,
            "observations": count,
            "covered_seconds": covered,
        }

    findings = []
    for row in gpu_rows:
        if row["coverage_fraction"] < 0.9:
            findings.append(
                {
                    "gpu": row["gpu"],
                    "kind": "insufficient_activity_coverage",
                    "action": "Check exporter field support, mapping and scrape failures.",
                }
            )
        if (row["low_activity_with_demand_seconds"] or 0) > 0:
            findings.append(
                {
                    "gpu": row["gpu"],
                    "kind": "low_activity_with_demand",
                    "action": "Trace CPU scheduling, batch formation and synchronization; "
                    "replay matched load before changing deployment code.",
                }
            )
        if (row["idle_seconds_no_demand"] or 0) > 0:
            findings.append(
                {
                    "gpu": row["gpu"],
                    "kind": "idle_without_demand",
                    "action": "Investigate demand, routing or capacity sizing; "
                    "this alone is not evidence of an inference-code bottleneck.",
                }
            )
    return {
        "schema_version": 1,
        "deployment": config.deployment,
        "engine": config.engine,
        "config_sha256": config.sha256,
        "metadata": config.metadata,
        "allocation": config.allocation,
        "measurement": "sampled engine inactivity estimate, not a CUDA kernel-gap trace",
        "integration": "right-sample weighting over valid adjacent scrape intervals",
        "attribution": "Declared exclusive worker GPUs"
        if config.allocation == "exclusive"
        else "Shared GPU correlation only; no ownership attribution",
        "frames": len(frames),
        "scrape_errors": {
            source: sum(bool(f[source].error) for f in sources) for source in ("gpu", "serving")
        },
        "gpus": gpu_rows,
        "serving": serving,
        "findings": findings,
        "timeline": timeline,
        "limitations": [
            "DCGM watch cadence/averaging may differ from scrape cadence; "
            "subinterval gaps are invisible.",
            "No source timestamp means exporter freshness cannot be proven.",
            "Zeros from counter multiplexing can resemble idle; "
            "verify DCGM metric groups on hardware.",
            "Demand transitions and excessive endpoint skew remain unclassified.",
            "Low tensor activity alone is not evidence of waste in memory-bound decoding.",
            "Observations suggest tests, not causes or permission to change production.",
        ],
    }
