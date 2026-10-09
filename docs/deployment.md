# Deployment guide

Start with the engine-specific quickstarts in the README. Obsrv runs beside an existing
serving process, in its own environment or as a sidecar. It never launches/restarts the
engine or injects a profiler into it.

## Activation and resuming collection

```bash
obsrv activate --engine vllm --url http://127.0.0.1:8000 \
  --deployment coding --revision YOUR_CODE_COMMIT_OR_IMAGE_DIGEST \
  --trace-dir /shared/profiles
```

For SGLang use `--engine sglang` and its server URL. The generated configuration is
stored under `.obsrv/<deployment-name-and-hash>/obsrv.json` in the activation directory.
The activation output includes its absolute path and subsequent commands.

Use `--setup-only` to configure without collecting, or `--count 2` to test a finite
number of scrapes. Activation only discovers safe, allowlisted metadata from the engine.
If model discovery is unavailable or advertises several models, supply `--model`.
Stop foreground collection with Ctrl+C. Resume it with:

```bash
obsrv collect --deployment coding
```

Named deployments are resolved from the current working directory. Use `--config`
with the generated absolute path from another directory or after `--state-dir` overrides.
Configuration is reloaded between scrapes; update the deployed revision/configuration
at rollout so observation windows split into distinct identity cohorts. Activation
refuses to overwrite an existing configuration with conflicting identity information.

## Authentication and admin access

Use an environment reference rather than a secret argument:

```bash
obsrv activate --engine vllm --url https://SERVING_ADMIN_HOST \
  --deployment coding --auth-env SERVING_ADMIN_TOKEN --trace-dir /shared/profiles
```

Populate that environment variable through your existing secret-management mechanism.
Obsrv attaches it as a Bearer token only to its serving endpoint. It does not persist
its value, include it in exports, or follow HTTP redirects. URLs cannot contain credentials,
query strings or fragments. The same environment variable must be available when profiling.

Point at one deployment's internal admin endpoint, not a public load balancer that could
route start and stop to different replicas. Native profile endpoints have no obsrv ownership
token. Reserve profile control for obsrv during captures and share one state/lock volume
between collector/controller processes for that deployment. In separate containers,
configure a private network/address reachable by the collector; loopback only works
when the serving process and collector share a network namespace.

## Trace volumes and workers

The engine must write traces to a volume readable by obsrv. vLLM writes to the absolute
`torch_profiler_dir` configured at startup. For SGLang the controller requests a unique
capture subdirectory under `server_trace_dir` (or `trace_dir` if the paths are identical).
Set `server_trace_dir` when the server and sidecar mount the same volume at different paths.

Discovery cannot verify expected distributed ranks. The initial identity has
`workers=["unattributed"]` and `configuration.topology_verified=false`. Useful GPU traces
are retained, but `profile` reports partial coverage and exits with code 2.

For known topology, edit the config with explicit worker names, safe relative trace globs
and `topology_verified=true`. Inspect actual filenames; do not guess rank mapping. The
[advanced template](../examples/deployment.toml) shows a two-rank SGLang example:

```toml
[trace_patterns]
rank0 = "**/*TP-0*.trace.json.gz"
rank1 = "**/*TP-1*.trace.json.gz"
```

Each trace must map to one worker. Unmatched, ambiguous, invalid and CPU-only traces
limit coverage. The generic activation path groups traces as unattributed; it does not
claim single-GPU or distributed completeness. CUDA graphs can restrict operator attribution;
inspect VibeSys trace certification instead of automatically disabling production graphs.

## Capture lifecycle

```bash
obsrv profile --deployment coding --duration 30
```

Capture proceeds through starting, recording, stopping/flushing and completion. The
active window defaults to 30 seconds, with a maximum of 300. Flush timeout is independently
configurable (default 900 seconds); native flush can substantially outlast the recording
window and disrupt serving. All of that period is marked diagnostic.

If the start is definitively rejected, obsrv does not stop a potentially foreign profiler.
An ambiguous start/stop retains durable state and blocks another capture. After checking
server identity and reserving control, stop the owned session explicitly:

```bash
obsrv profile --deployment coding --recover CAPTURE_ID
```

Recovery requires the original endpoint and identity configuration. It does not claim
traces are complete; import recovered artifacts explicitly. A hard process crash cannot
guarantee the engine stops profiling. Use service supervision and inspect `report`.

## Nsight Systems and manual import

```bash
obsrv import --deployment coding --profiler torch --worker rank0 trace.json.gz
obsrv import --deployment coding --profiler nsys --worker rank0 profile.sqlite
obsrv import --deployment coding --profiler nsys --worker rank0 profile.nsys-rep
```

VibeSys's default CUDA serving profiler is Nsight Systems. This release imports its
captures; automatic live captures use the supported PyTorch path. It does not attach
Nsight to arbitrary running processes or launch an Nsight-wrapped production server.
`.nsys-rep` conversion needs the NVIDIA `nsys` CLI; SQLite import does not. The imported
trace must belong to the declared deployment/worker. Import time is not capture time.

## Optional continuous sources

Serving Prometheus metrics are supplemental: latency/TTFT/TPOT/ITL interval histograms,
request/token rates and queue/cache demand where exposed. Missing/reset/changed/stale
series remain unavailable. Histogram estimates are not exact per-request percentiles,
and an unbounded tail is not replaced with an invented finite value. Some engines do
not expose an error counter, so error rate can be unavailable.

DCGM is off by default. Configure its URL plus explicit physical/MIG device ownership:

```toml
dcgm_url = "http://127.0.0.1:9400/metrics"
gpu_selectors = [{UUID = "GPU-REPLACE-WITH-REAL-UUID"}]
```

Add `GPU_I_ID` for a MIG instance. Overlapping ownership selectors are rejected. A serving
Bearer token is not sent to the DCGM endpoint; use a separately secured exporter/network.

## Storage and service operations

Defaults are a 5-second scrape interval, 15-second maximum interval gap, 7-day retention
and 5-GiB local data quota. Records live in SQLite/WAL; artifacts are immutable SHA256
copies. Export bundles are preserved and verified against their manifests.

On quota exhaustion obsrv pauses new data writes and records a storage-limit event.
Remove/archive exports deliberately or expand capacity; it does not silently discard them.
Manage the serving engine's trace volume separately: native trace output is outside the
collector's storage quota. The control database has small bookkeeping overhead beyond it.

Use a service manager or sidecar lifecycle for foreground collection. No daemon,
cloud resources, optimization-agent calls or uploads are installed automatically.
