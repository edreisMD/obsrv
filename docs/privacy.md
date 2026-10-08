# Privacy and data handling

## What stays local

Obsrv fetches metadata and metrics from operator-configured serving/exporter endpoints
and sends native profile-control requests. It submits no inference prompts/completions.
It does not upload data to an agent, cloud service or external analytics endpoint.

Local deployment configuration contains the serving URL, model/deployment identity,
trace/state paths and authentication environment-variable name. It excludes the token
value and is created with mode 0600. Credentials in URLs are rejected. Server configuration
is allowlisted rather than copied wholesale; scrape/capture errors retain exception types,
not server response bodies. Serving credentials are not reused for optional DCGM requests.

Evidence exports contain declared deployment/model/hardware identities, selected metric
labels, profiler summaries and exact raw traces. Those identities and labels can themselves
be confidential even when they are not authentication secrets.

**Raw traces are not automatically redacted.** They can contain names, paths, stack or
annotation metadata emitted by the engine. Preserving exact bytes permits integrity
verification and reproducible analysis, but does not make an export safe to share.
Review artifacts and model/deployment identifiers before giving them to a third party.

## Repository and package publication

Publish source, synthetic fixtures and documentation only. Do not commit:

- `.obsrv/`, `state/`, `artifacts/`, `traces/`, `exports/`, logs or databases;
- generated `obsrv.json`, real deployment configs or private model/host identifiers;
- `.env` files, private keys, credentials, virtual environments or caches;
- actual profiler traces, serving requests, experiment outputs or private Git history.

`.gitignore` excludes common runtime files. Build configuration also explicitly selects
source-distribution contents and excludes sensitive runtime formats. The release checker
inspects source candidates and built archives, rejects unexpected files and flags private
home paths, credential URLs, key/token patterns and email addresses without printing values.

```bash
python scripts/check_release.py
python scripts/check_release.py --archives dist/*.whl dist/*.tar.gz
```

These checks reduce accidental leakage; they do not prove that arbitrary text has no
confidential information. Review the exact release inventory and Git history before publishing.
No CI job is configured to publish releases automatically. GitHub Actions needs no private
tokens from the project and has read-only repository permissions.

## Reporting a sensitive issue

Do not post actual credentials, raw production traces or private request data in public
issues. Reproduce with minimal synthetic data and describe the affected behavior. Follow
your deployment's incident/secret-rotation process if data has already been exposed.
