"""Union observed GPU activity in Chrome/PyTorch traces, excluding CPU events."""

import math


def trace_gaps(document, *, device, start_us, end_us):
    """Return unobserved GPU-activity gaps within an explicitly selected trace window.

    Input follows Chrome trace microsecond timestamps. This is scoped to captured
    processes, not proof that a shared device was globally idle. Lost trace events
    or missing memcpy/NCCL processes can inflate gaps.
    """
    if not all(math.isfinite(x) for x in (start_us, end_us)) or end_us <= start_us:
        raise ValueError("Trace window must have finite start < end in microseconds")
    events = document.get("traceEvents", [])
    intervals = []
    counts = {"kernel": 0, "gpu_memcpy": 0, "gpu_memset": 0}
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in counts:
            continue
        args = event.get("args", {})
        if str(args.get("device")) != str(device):
            continue
        ts, duration = event.get("ts"), event.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(duration, (int, float)):
            raise ValueError("Selected GPU events require numeric ts/dur")
        if not math.isfinite(ts) or not math.isfinite(duration) or duration < 0:
            raise ValueError(
                "Selected GPU events require finite timestamps and nonnegative durations"
            )
        start, end = max(start_us, ts), min(end_us, ts + duration)
        if start < end:
            intervals.append((start, end))
            counts[category] += 1
    if not intervals:
        raise ValueError("No GPU events for selected device/window; cannot infer an idle device")
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    cursor = start_us
    gaps = []
    for start, end in merged:
        if cursor < start:
            gaps.append([cursor, start])
        cursor = end
    if cursor < end_us:
        gaps.append([cursor, end_us])
    window = (end_us - start_us) / 1e6
    gap_seconds = sum(end - start for start, end in gaps) / 1e6
    return {
        "schema_version": 1,
        "measurement": "gaps in captured GPU activity timeline",
        "device": str(device),
        "window_seconds": window,
        "gap_seconds": gap_seconds,
        "gap_fraction": gap_seconds / window,
        "event_counts": counts,
        "gaps_us": gaps,
        "busy_intervals_us": merged,
        "limitations": [
            "Scoped to captured processes; not global hardware idle time.",
            "Requires complete GPU traces; does not validate dropped events.",
            "Trace overhead must be measured against an untraced control.",
        ],
    }
