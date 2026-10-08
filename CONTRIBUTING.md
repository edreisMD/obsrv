# Contributing

Obsrv is an experimental production profiling/evidence package. Changes should keep
profiler evidence distinct from supplementary serving metrics, and preserve provenance,
partial coverage and unavailable measurements.

## Local setup

From this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . --group dev
python -m pytest
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python scripts/check_release.py
```

Alternatively, `uv sync --locked` installs the package and development tools.
Loopback HTTP tests need permission to bind a temporary local port. No GPU/model
download is needed for the regular suite. Synthetic fixtures do not establish real
serving compatibility; see docs/nvidia-validation.md before making that claim.

Tests must pass from an isolated checkout. Optional reference-source parity tests can
use `VIBESYS_REFERENCE_ROOT` pointing at the pinned upstream checkout; they skip if
that separate checkout is absent. The regular suite also checks packaged analyzer
hashes and synthetic reference measurements.

## Changes and reviews

- Include a regression test for behavior changes, especially identity boundaries,
  capture ownership, unavailable data and artifact integrity.
- Keep pure analysis separate from HTTP, file and process I/O.
- Never claim trace certification establishes model correctness or official acceptance.
- Changes to vendored analyzers need explicit provenance updates and parity validation.
- Use synthetic, minimal fixtures. Never submit real traces, prompts, host information,
  deployment configs, environment files, credentials or private model identifiers.
- Do not paste server responses or authentication details into issues or CI logs.

The CI workflow uses read-only repository permissions, runs no optimization agents
and publishes no releases. Follow docs/releasing.md before publishing artifacts.
