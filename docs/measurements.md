# Measurement reference

## GPU and serving metrics

The GPU source is NVIDIA DCGM Exporter. Request queues, token counters and
histograms come from a vLLM or SGLang Prometheus endpoint. No request bodies or
prompts are collected. All labels on selected metric families are retained.

- Engine inactivity: `(1 - engine_active) × interval`, summed across adjacent
  valid scrapes. This is graphics/compute engine inactivity, not all GPU engines
  or unused FLOPs. Current values weight the preceding scrape interval; the
  hardware watch window may differ.
- No-demand inactivity: correlated with zero running and waiting requests at
  both interval ends. Consider capacity, routing and demand.
- With-demand inactivity: correlated with nonzero demand at both interval ends.
  Investigate scheduling, batching, host work and synchronization with a trace.
- Low activity with demand: seconds under the configured activity threshold.
  This differs from the fractional inactivity estimate.
- SM activity: averaged SM resource use, kept separate from temporal engine
  inactivity. Low tensor activity alone is not evidence of waste in memory-bound
  autoregressive decoding.
- Counter rates: adjacent per-series deltas. Resets, missing series and failures
  are skipped. No assumed zero during an outage.
- Latency: paired histogram sum/count deltas within matching valid intervals.
  These are server-side means, not p95/p99 or client SLO measurements.

Missing, nonfinite, sentinel, stale timestamped and ambiguous values stay unknown.
Untimestamped exporter freshness cannot be established. Stable demand joins
require closely aligned acquisition envelopes. First/last samples are not
extrapolated. Supported profiling counters and multiplexing vary by hardware;
check with `dcgmi profile -l -i 0` before interpreting zero values.

Select one serving scheduler/engine using actual exported labels. Duplicate
queue gauges are rejected rather than summed. Declare every tensor-parallel GPU
once. For MIG, select distinct instances, never parent and children together.
Shared GPUs support correlation only, not workload ownership attribution.

## Fine timing

Trace analysis clips and unions observed GPU kernel, memcpy and memset intervals
for one explicitly selected device/window. CPU events are excluded. Concurrent
kernels are not double counted. Gaps are scoped to captured processes; missing
processes or dropped events can inflate them. An empty GPU trace is an error,
not proof of 100% idle. Trace overhead must be compared with an untraced control.

## Apple telemetry

The optional local monitor reads the driver's `Device Utilization %` and GPU
memory counters from `ioreg`. These are best-effort, system-wide counters with
undocumented sampling semantics. No idle conversion or experiment attribution
is inferred. Unsupported or inaccessible counters remain unavailable. It polls
every two seconds; changing the GPU load is never part of collection.

## Storage and dashboard

SQLite uses WAL and commits each frame. Schema v1 stores a run manifest, SHA256,
status and frames with acquisition timestamps, samples and failure types.
JSONL exports begin with the manifest. `null` means unknown.

The dashboard is read-only with respect to benchmark artifacts and capture
files. It binds to 127.0.0.1 and exposes only bundled assets and normalized local
state. No arbitrary file browsing or remote control endpoints are provided.
It is not an authenticated fleet monitoring service. Reporting loads captures
into memory; rotate long captures and use an existing time-series database for
fleet retention.
