"""Production evidence exports; a future VibeSys importer owns trust and context injection."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis import PROVENANCE
from .models import Identity, canonical
from .store import EvidenceStore

MAX_CONTEXT_CHARS = 16_384


@dataclass(frozen=True)
class ProductionEvidenceBundle:
    """Stable v1 boundary; metrics/artifact digests follow VibeSys conventions."""

    identity: dict[str, Any]
    identity_id: str
    observation_period: dict[str, float | None]
    observations: list[dict]
    profiles: list[dict]
    captures: list[dict]
    artifacts: list[dict]
    limitations: list[str]
    collection_events: list[dict]
    generated_at: float
    profiler_provenance: dict
    schema: str = "obsrv.production-evidence"
    version: int = 1
    source: str = "production_observations"
    correctness: str = "not_evaluated"
    official_gate_acceptance: str = "not_evaluated"

    def record(self):
        from dataclasses import asdict

        return asdict(self)


def build_bundle(store: EvidenceStore, identity: Identity):
    key = identity.fingerprint
    observations = sorted(store.rows("observations", key), key=lambda r: r["end"])
    profiles = sorted(store.rows("profiles", key), key=lambda r: r["timestamp"])
    captures = sorted(store.rows("captures", key), key=lambda r: r["created"])
    # Record diagnostic overlap at export too: a capture may start during a scrape.
    for observation in observations:
        observation["diagnostic"] = observation["diagnostic"] or store.diagnostic_overlap(
            identity.deployment, observation["start"], observation["end"]
        )
    artifacts = {}
    limitations = [
        "Production observations guide hypotheses; they do not establish causal efficiency gains.",
        "Correctness and VibeSys official gate acceptance were not evaluated.",
        "Profiler captures perturb serving; they are not unprofiled latency benchmarks.",
        "Identity and worker ownership are operator-declared; obsrv does not attest deployed code.",
    ]
    seen_workers = set()
    for profile in profiles:
        for artifact in profile["artifacts"]:
            artifacts[artifact["digest"]["value"]] = artifact
        limitations.extend(profile["limitations"])
        if profile["status"] == "observed":
            seen_workers.add(profile["worker"])
        if not profile["capture_id"]:
            limitations.append(
                "Imported trace has no obsrv capture period; import time is not capture time."
            )
    if not identity.configuration.get("topology_verified", True):
        limitations.append(
            "Worker topology unverified; imported traces do not prove complete deployment coverage."
        )
    events = store.rows("events")
    if any(event.get("event_kind") == "storage_limit" for event in events):
        limitations.append(
            "Local storage quota interrupted collection/import; inspect collection_events."
        )
    missing = set(identity.workers) - seen_workers
    if missing:
        limitations.append("No observed GPU profile for workers: " + ", ".join(sorted(missing)))
    for observation in observations:
        limitations.extend(observation["coverage"])
    for capture in captures:
        limitations.extend(capture["limitations"])
        if not capture["cleanup_confirmed"]:
            limitations.append("Capture cleanup requires operator recovery: " + capture["id"])
    starts = [o["start"] for o in observations] + [c["created"] for c in captures]
    ends = [o["end"] for o in observations] + [c.get("finished", c["updated"]) for c in captures]
    bundle = ProductionEvidenceBundle(
        identity=identity.record(),
        identity_id=key,
        observation_period={
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
        },
        observations=observations,
        profiles=profiles,
        captures=captures,
        artifacts=list(artifacts.values()),
        limitations=list(dict.fromkeys(limitations)),
        collection_events=events,
        generated_at=time.time(),
        profiler_provenance=PROVENANCE,
    )
    return bundle


def context_summary(bundle: ProductionEvidenceBundle):
    identity = bundle.identity
    parts = [
        "# obsrv production profiling evidence\n",
        f"Deployment: {identity['deployment']} | revision: {identity['revision']}\n",
        f"Engine: {identity['engine']} {identity['engine_version']} | model: {identity['model']}\n",
        f"Identity SHA256: {bundle.identity_id}\n",
        "Source: production observations. Correctness and gate acceptance: not evaluated.\n",
        "Primary evidence: pinned VibeSys profiler analysis. Serving/DCGM are supplemental.\n",
        "Full records and exact profiler reports: bundle.json; hashed traces: artifacts/.\n",
        "## Coverage and interpretation\n",
        *(f"- {item}\n" for item in bundle.limitations),
    ]
    # Newest profiles are most useful for the next hypothesis; retain evidence IDs.
    for profile in reversed(bundle.profiles):
        parts.append(
            f"\n## Profile {profile['id']} | {profile['profiler']} | "
            f"worker {profile['worker']} | {profile['status']}\n"
        )
        parts.append(
            "Artifact hashes: "
            + ", ".join(a["digest"]["value"] for a in profile["artifacts"])
            + "\n"
        )
        for name, report in profile["reports"].items():
            parts.append(f"### VibeSys {name}\n{report}\n")
    if bundle.observations:
        # Do not misrepresent a mean of interval p95s as a deployment p95.
        latest = bundle.observations[-1]
        parts.append(
            f"\n## Latest supplemental interval {latest['start']}–{latest['end']} "
            f"(diagnostic={latest['diagnostic']})\n"
        )
        for metric in latest["metrics"]:
            value = metric["value"]
            if value is not None:
                parts.append(
                    f"- {metric['name']}: {value:g} {metric['unit']} [{metric['source']}]\n"
                )
    text = "".join(parts)
    if len(text) > MAX_CONTEXT_CHARS:
        suffix = "\n[Context truncated; read bundle.json for complete reports and limitations.]\n"
        text = text[: MAX_CONTEXT_CHARS - len(suffix)] + suffix
    return text


def export_context(store: EvidenceStore, identity: Identity) -> Path:
    bundle = build_bundle(store, identity)
    return store.save_export(bundle.record(), context_summary(bundle))


def report(store: EvidenceStore, identity: Identity | None = None):
    identities = [identity.record()] if identity else store.rows("identities")
    versions = []
    for raw in identities:
        raw["workers"] = tuple(raw["workers"])
        item = Identity(**raw)
        bundle = build_bundle(store, item)
        versions.append(
            {
                "identity_id": item.fingerprint,
                "identity": item.record(),
                "observation_period": bundle.observation_period,
                "observation_count": len(bundle.observations),
                "profiles": bundle.profiles,
                "captures": bundle.captures,
                "latest_supplemental_interval": bundle.observations[-1]
                if bundle.observations
                else None,
                "limitations": bundle.limitations,
            }
        )
    return {
        "versions": versions,
        "comparison_policy": "Versions are separate production cohorts, not paired benchmarks.",
    }


def verify_export(path: str | Path):
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for name in ("bundle", "context"):
        filename = "bundle.json" if name == "bundle" else "context.md"
        if EvidenceStore.hash_file(root / filename) != manifest[name + "_sha256"]:
            raise ValueError(f"{name} digest mismatch")
    bundle = json.loads((root / "bundle.json").read_text())
    if bundle["schema"] != "obsrv.production-evidence" or bundle["version"] != 1:
        raise ValueError("unsupported production evidence schema")
    if canonical(bundle["artifacts"]) != canonical(manifest["artifacts"]):
        raise ValueError("artifact manifests disagree")
    for artifact in manifest["artifacts"]:
        target = (root / artifact["path"]).resolve()
        if root not in target.parents or not artifact["path"].startswith("artifacts/"):
            raise ValueError("unsafe artifact path")
        if EvidenceStore.hash_file(target) != artifact["digest"]["value"]:
            raise ValueError("artifact digest mismatch")
    return True
