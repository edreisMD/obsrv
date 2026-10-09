# First cluster trial and VibeSys handoff

This is an experimental alpha. Local end-to-end tests exercise real loopback HTTP
and installed CLI processes against synthetic vLLM/SGLang protocol fixtures. They do
not launch CUDA engines or establish production overhead. Start with a test replica.

## Install and attach

Install from GitHub into a separate Python 3.11+ environment on the serving
host or a sidecar. No PyTorch dependency is needed in this collector environment.

```bash
python3 -m venv .obsrv-venv
source .obsrv-venv/bin/activate
python -m pip install "git+https://github.com/edreisMD/obsrv.git@main"
obsrv --help
```

For vLLM, enable the native profiler **through your normal deployment process**:

```bash
vllm serve "$MODEL" --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/obsrv-profiles","torch_profiler_record_shapes":true,"torch_profiler_with_memory":true,"torch_profiler_with_stack":false}'
```

This uses vLLM's >=0.13 configuration interface. If an existing server was launched
without profiling, passive collection still works; obsrv cannot enable its profiler
by itself. Use the [README](../README.md) for the complete SGLang launch example.

Mount `/tmp/obsrv-profiles` into the collector too. Use the specific replica's private
API URL, rather than a load balancer that could route start and stop to different
servers. For a multi-node deployment, arrange shared storage for all expected traces.
Keep state on a persistent volume and run one collector per deployment ownership scope.

```bash
obsrv activate --engine vllm --url http://127.0.0.1:8000 \
  --deployment coding --revision YOUR_IMAGE_OR_CODE_DIGEST \
  --trace-dir /tmp/obsrv-profiles
```

This discovers the model/version and collects passively in the foreground. Leave it
running. In another terminal in the **same working directory/environment**:

```bash
obsrv profile --deployment coding --duration 30
obsrv report --deployment coding
obsrv export-context --deployment coding
```

Keep representative requests flowing during capture; obsrv does not generate traffic.
Profile flushing can take minutes. The default flush timeout is 900 seconds, separate
from the 30-second active window. Profiling perturbs serving performance.

By default, worker topology is unknown: `profile` can return exit code **2** with a
useful `partial` capture. Inspect its JSON rather than interpreting this as an install
failure. Configure the true `identity.workers`, `configuration.topology_verified` and
`trace_patterns` in the printed config before claiming complete distributed coverage.
Use real rank-qualified filenames, not guessed glob patterns. See
[deployment.md](deployment.md) and [nvidia-validation.md](nvidia-validation.md).
For authentication use `--auth-env YOUR_TOKEN_ENV_NAME`; no token values belong in configs.

## Feed the optimization loop

The export command prints a directory containing `context.md`, `bundle.json`,
`manifest.json` and `artifacts/`. Copy that whole directory and verify it:

```bash
obsrv verify /path/to/export
```

Give `context.md` to the agent as **production observations for hypothesis planning**,
with `bundle.json` and artifacts available for deeper inspection. Keep limitations,
deployment revision, capture period and worker coverage attached. This release has
**no automatic VibeSys hypothesis importer**; do not pass `bundle.json` to APIs that
expect VibeSys `TrustedEvidence`, and do not mark it accepted experiment evidence.

To analyze exported PyTorch bytes directly using VibeSys's existing analyzer, restore
the format suffix on its content-addressed artifact filename:

```bash
export OBSRV_EXPORT=/path/to/export
python - <<'PY'
import json, os, shutil
from pathlib import Path
root = Path(os.environ["OBSRV_EXPORT"])
bundle = json.loads((root / "bundle.json").read_text())
profile = next(p for p in bundle["profiles"] if p["profiler"] == "torch")
artifact = profile["artifacts"][0]
suffix = ".json.gz" if artifact["format"] == "chrome_trace_json_gzip" else ".json"
target = Path("production-trace" + suffix)
shutil.copyfile(root / artifact["path"], target)
print(target)
PY
python "$VIBESYS_ROOT/resources/profilers/torch/analyze_torch_profile.py" summary production-trace.json.gz
```

Use `production-trace.json` for an uncompressed artifact. Analyzer parity is tested
against the pinned VibeSys source revision listed in `bundle.json`; a friend's different
revision may produce different reports. Nsight SQLite imports are supported separately.

Review raw traces before sharing: they are not redacted and can contain application
metadata. Obsrv does not upload them. Controlled VibeSys correctness/performance
experiments remain necessary before accepting an optimization.

## Short message to send

> I've built obsrv, an alpha production profiler for vLLM/SGLang that reuses VibeSys's
> analyzers. Install the attached wheel in a separate Python 3.11+ environment. Follow
> this guide to enable native profiling and share the trace directory, then run
> `obsrv activate --engine vllm --url YOUR_REPLICA_URL --deployment coding --revision YOUR_DIGEST --trace-dir YOUR_TRACE_DIR`.
> While requests are flowing, run `obsrv profile --deployment coding --duration 30`
> and `obsrv export-context --deployment coding` from another terminal in the same directory.
> The export contains agent context, full reports and hashed traces. Use it as production
> planning context in your VibeSys loop; automatic hypothesis import isn't wired yet.
> Local installed-CLI tests pass for both protocols, but please trial it on a test GPU
> replica first and check rank coverage/overhead before using production captures.
