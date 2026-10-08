"""One-command attachment and engine-specific deployment settings."""

import json
import os
import re
from pathlib import Path

from .config import Config, endpoint
from .http import HttpTransport
from .models import Identity, digest


def deployment_settings(engine: str, trace_dir: str | Path):
    """Return launch settings, without launching or altering the model process."""
    path = str(Path(trace_dir).resolve())
    if engine == "vllm":
        settings = {
            "profiler": "torch",
            "torch_profiler_dir": path,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": True,
            "torch_profiler_with_stack": False,
        }
        return {
            "engine": engine,
            "args": ["--profiler-config", json.dumps(settings)],
            "env": {},
            "note": "vLLM >=0.13 profiler-config; profiling starts only on operator command.",
        }
    if engine == "sglang":
        return {
            "engine": engine,
            "args": ["--enable-metrics"],
            "env": {"SGLANG_TORCH_PROFILER_DIR": path},
            "note": "obsrv sends CPU/GPU and record_shapes=true in each native capture request.",
        }
    raise ValueError("engine must be vllm or sglang")


def deployment_directory(deployment: str):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", deployment)[:64] + "-" + digest(deployment)[:8]
    return (Path(".obsrv") / safe_name).resolve()


def activate(
    *,
    engine: str,
    url: str,
    deployment: str | None = None,
    revision: str = "unknown",
    model: str | None = None,
    state_dir: str | Path | None = None,
    trace_dir: str | Path | None = None,
    auth_env: str | None = None,
    transport=None,
) -> tuple[Config, Path, dict]:
    """Discover safe metadata and write a deployment config; never enables profiling itself."""
    endpoint(url)
    deployment = deployment or engine + "-" + digest(url)[:8]
    transport = transport or HttpTransport(auth_env)
    notices = []

    def read(path):
        try:
            data = json.loads(transport.get(url.rstrip("/") + path, 5))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            notices.append(f"metadata discovery {path} unavailable: {type(exc).__name__}")
            return {}

    models = read("/v1/models")
    choices = [
        m["id"]
        for m in models.get("data", [])
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    ]
    if model is None:
        if len(choices) != 1:
            raise ValueError("model discovery ambiguous/unavailable; supply --model")
        model = choices[0]
    elif choices and model not in choices:
        raise ValueError("selected model is not advertised by this deployment")
    info = read("/version" if engine == "vllm" else "/get_server_info")
    version = str(info.get("version") or "unknown")
    settings = {
        "topology_verified": False,
        "identity_source": "discovered_metadata",
        "model_name": model,
    }
    # Store only an allowlist of server args; never copy the server's entire configuration.
    server_args = info.get("server_args") or {}
    for key in ("tp_size", "dp_size", "pp_size", "dtype", "attention_backend"):
        value = server_args.get(key) if isinstance(server_args, dict) else None
        if isinstance(value, (int, str, bool)):
            settings[key] = value
    identity = Identity(
        deployment,
        revision,
        engine,
        version,
        model,
        settings,
        {"source": "trace_device_properties_when_available"},
        ("unattributed",),
    )
    root = Path(state_dir or deployment_directory(deployment)).resolve()
    trace = Path(trace_dir or root / "traces").resolve()
    config = Config(
        identity, url, trace, root, serving_labels={"model_name": model}, auth_env=auth_env
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / "obsrv.json"
    if path.exists():
        old = Config.load(path)
        if old.identity.fingerprint != config.identity.fingerprint or old.serving_url != url:
            raise ValueError(
                "activation config already exists with a different identity; use a new state-dir"
            )
        return (
            old,
            path,
            {"notices": notices, "deployment_settings": deployment_settings(engine, trace)},
        )
    raw = {
        "identity": identity.record(),
        "serving_url": url,
        "trace_dir": str(trace),
        "state_dir": str(root),
        "serving_labels": config.serving_labels,
    }
    if auth_env:
        raw["auth_env"] = auth_env
    # Exclusive creation protects existing deployment attribution/configuration.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(raw, file, indent=2)
        file.write("\n")
    path.chmod(0o600)
    notices.append("Worker topology is unverified; traces remain unattributed until configured.")
    if revision == "unknown":
        notices.append(
            "Revision unknown; supply --revision with a code commit or container digest."
        )
    return (
        config,
        path,
        {"notices": notices, "deployment_settings": deployment_settings(engine, trace)},
    )
