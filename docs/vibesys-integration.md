# Future VibeSys integration boundary

Obsrv exports `obsrv.production-evidence` version 1, a `ProductionEvidenceBundle`.
It is not the existing candidate-stage `TrustedEvidence` contract. No VibeSys files
or hypothesis-generation paths are modified in this release.

A future VibeSys importer should:

1. Verify the export manifest and artifact hashes, schema version and deployment identity.
2. Bind the deployed revision/config/model/hardware to the candidate being optimized.
   Obsrv's identity fingerprint hashes the declaration, not the deployed executable.
3. Establish workload/time/worker coverage, account for profile certification failures,
   and keep diagnostic capture windows separate from unprofiled serving performance.
4. Present `context.md` and relevant exact analyzer reports/artifacts as production
   planning guidance. Preserve the original observation period, evidence ID and digest.
5. Run controlled accuracy and performance stages for the new hypothesis. Those stages,
   not the production bundle, produce official accepted evidence and accepted rounds.

Field correspondence:

| obsrv | VibeSys-compatible concept | Importer responsibility |
| --- | --- | --- |
| Metric name/value/unit/direction | `EvidenceMetric` | Validate finite values; handle unavailable values separately |
| Artifact relative path + SHA256 digest | `ArtifactDigest` / `ContentDigest` | Verify bytes and safe paths |
| Identity fingerprint + declaration | Candidate/environment identity hints | Bind to actual code snapshot and environment; do not equate hashes |
| Exact reports, compact context | Semantic summary / planning guidance | Select relevant records within VibeSys's 16,384-character summary budget |
| Observed/unsupported profile status | Profiler observation capabilities | Preserve missing fields, certification and partial coverage |
| Correctness/gate status `not_evaluated` | No accepted experiment fact | Never invent an accepted round or pass result |

The stable SDK boundary is `build_bundle`, `export_context`, and `verify_export`.
`bundle.json` retains complete records even when `context.md` is truncated. Profiles
retain analyzer provenance and worker labels. Arbitrary imported traces have an import
timestamp, not an inferred production capture timestamp. Trace-only bundles therefore
have a null observation period unless an obsrv capture or passive interval establishes it.

A runnable manual handoff is in [friend-quickstart.md](friend-quickstart.md). Artifact
`format` identifies compressed Chrome JSON, plain Chrome JSON, Nsight reports or
SQLite without exposing original filenames. Restore `.json.gz` for compressed traces
before using the upstream analyzer, whose reader selects decompression by suffix.
