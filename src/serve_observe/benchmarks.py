"""Read paired benchmark artifacts without executing experiment code or trusting scores."""

import hashlib
import json
import math
import statistics
from pathlib import Path

MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def paired_benchmark(document, *, name, observed_ns):
    """Accept the documented records schema and recompute comparisons from timings.

    A producer's 'passed' flag remains a producer claim, not independently attested
    correctness. Repetitions/cases are distinct from independent experiment runs.
    """
    if (
        not isinstance(document, dict)
        or document.get("mode") != "benchmark"
        or not isinstance(document.get("records"), list)
    ):
        raise ValueError("Expected mode=benchmark and paired records")
    rows = document["records"]
    if not rows or len(rows) > 10000:
        raise ValueError("Expected 1..10000 paired trials")
    reference, candidate, tokens, trial_rows = [], [], [], []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Expected a paired trial object")
        values = (row.get("reference_s"), row.get("candidate_s"))
        if any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
            for v in values
        ):
            raise ValueError("Paired timings must be finite and positive")
        if not math.isfinite(values[0] / values[1]) or values[0] / values[1] <= 0:
            raise ValueError("Paired timing ratio is outside the supported numeric range")
        token_count = row.get("output_tokens_including_eos", row.get("output_tokens"))
        if token_count is not None and (
            isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0
        ):
            raise ValueError("Output token counts must be nonnegative integers")
        reference.append(values[0])
        candidate.append(values[1])
        tokens.append(token_count)
        trial_rows.append(
            {
                "trial": index + 1,
                "case": str(row.get("case", "request"))[:120],
                "reference_s": values[0],
                "candidate_s": values[1],
                "speedup": values[0] / values[1],
            }
        )
    if any(not math.isfinite(sum(v)) for v in (reference, candidate)):
        raise ValueError("Total timing is outside the supported numeric range")
    token_units = {
        "output_tokens_including_eos" in r
        for r in rows
        if r.get("output_tokens_including_eos", r.get("output_tokens")) is not None
    }
    if len(token_units) > 1:
        raise ValueError("Use the same token count definition for all trials")
    speedup = math.exp(
        statistics.mean(math.log(r / c) for r, c in zip(reference, candidate, strict=True))
    )
    has_tokens = all(t is not None for t in tokens)
    metadata = {
        k: str(document[k])[:500]
        for k in ("scope", "model_revision", "candidate_sha256", "workload_sha256", "revision")
        if k in document
    }
    device = document.get("device", {})
    if not isinstance(device, dict):
        raise ValueError("Expected a device object")
    return {
        "id": hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
        "name": name,
        "observed_ns": observed_ns,
        "source": "paired_benchmark",
        "data_kind": "synthetic"
        if document.get("data_kind") == "synthetic"
        else "measured_artifact",
        "correctness": "reported_pass" if document.get("passed") is True else "unverified",
        "trial_count": len(rows),
        "speedup": speedup,
        "latency_reduction_fraction": 1 - sum(candidate) / sum(reference),
        "reference_median_s": statistics.median(reference),
        "candidate_median_s": statistics.median(candidate),
        "reference_tokens_per_second": sum(tokens) / sum(reference) if has_tokens else None,
        "candidate_tokens_per_second": sum(tokens) / sum(candidate) if has_tokens else None,
        "token_unit": "output tokens including EOS"
        if any("output_tokens_including_eos" in r for r in rows)
        else "output tokens",
        "hardware": str(device.get("device_name", "Not supplied"))[:200],
        "metadata": metadata,
        "trials": trial_rows,
        "notes": "Within-artifact paired comparison. File modification time orders history; "
        "it is not a verified experiment timestamp. No GPU idle measurement in this artifact.",
    }


def read_benchmarks(directory):
    runs, issues = [], []
    directory = Path(directory)
    if not directory.is_dir():
        return [], [{"kind": "benchmark_directory_unavailable"}]
    for path in sorted(directory.glob("*.json")):
        try:
            if path.is_symlink() or path.stat().st_size > MAX_ARTIFACT_BYTES:
                continue
            doc = json.loads(path.read_text())
            if not isinstance(doc, dict) or doc.get("mode") != "benchmark":
                continue
            runs.append(paired_benchmark(doc, name=path.stem, observed_ns=path.stat().st_mtime_ns))
        except (ValueError, TypeError, OSError, KeyError):
            issues.append({"kind": "invalid_benchmark_artifact", "name": path.name})
    return sorted(runs, key=lambda r: (r["observed_ns"], r["name"])), issues
