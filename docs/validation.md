# Validation status

Obsrv 0.1.0 is an experimental alpha. The local validation environment was macOS with
Python 3.14 and an isolated Python 3.11 environment. Test fixtures are synthetic protocol/trace examples, not production evidence.

Validated locally:

- Package installation, CLI/SDK operations and self-contained evidence exports.
- Loopback HTTP activation/discovery and native profiling flows for both engine protocols.
- Delayed flushing, cancellation, ownership, explicit recovery and partial worker coverage.
- Counter resets, identity rollouts, collection gaps and unavailable histogram observations.
- Artifact corruption detection, retention and export integrity.
- PyTorch summary/certification/report parity and Nsight runtime/gap parity against the
  pinned VibeSys source checkout; packaged hashes preserve analyzer provenance.
- Built wheel and source archive installation/content checks, including publication exclusions.

GitHub CI passed on Linux with Python 3.11–3.14, including package tests, release
audits and isolated installed-wheel CLI workflows. These runs require no GPUs or model downloads.
Optional source-reference parity tests skip when a separate upstream checkout is absent;
regular synthetic reference and packaged-provenance tests do not require it.

Not yet validated:

- Real NVIDIA vLLM/SGLang capture/CUPTI behavior across engine versions.
- Distributed rank attribution and capture coverage on actual deployment topologies.
- Collector overhead, profiler disruption or any production efficiency improvement.
- The future VibeSys hypothesis-loading integration.

Use [nvidia-validation.md](nvidia-validation.md) before adopting live profiling. Trace
certification, package tests and production observations are not model correctness tests
or VibeSys official gate acceptance.

## Publication preparation checks

- Python 3.14 with the pinned VibeSys reference checkout: **59 passed**.
- Python 3.11 in the sanitized standalone source bundle: **55 passed, 2 optional
  reference-source checks skipped**. No sibling checkout was needed.
- Lint and formatting passed in both environments.
- Wheel installation, CLI import and packaged provenance loaded from an isolated environment.
- Source/wheel/sdist inventories passed the publication audit. Synthetic runtime/config
  canaries were excluded from all three release archives.
- Serving authentication is separated from DCGM requests; the regression test confirms
  that the serving Bearer token is not forwarded to the exporter.
- The production profiler is published on GitHub’s `production-profiling` branch.
  The earlier dashboard remains on `main` pending the migration pull request.
  PyPI publication is separate and has not occurred.

The audit flags known private-data patterns and unexpected files. It is not a guarantee
that arbitrary future additions cannot contain confidential information; review each release.

## Cluster handoff checks

- Full local suite: **59 passed**, including upstream PyTorch/Nsight parity.
- Full suite against a fresh installed Python 3.11 wheel: **59 passed**.
- Fresh wheel installed in a separate Python 3.11 environment; both engine CLI
  workflows and exported-trace analysis exercised without source-tree imports.
- Real HTTP fixture flows: discovery, deployment-name selection, interval token
  rates/latency histograms, explicit capture start/stop, gzip trace analysis, unknown
  topology reported as partial, explicit rank coverage reported as completed, report,
  export, recipient-side verification, manual trace import and tamper rejection.
- Exported compressed PyTorch bytes are readable by the unchanged VibeSys analyzer
  after restoring the compression suffix; summaries match exactly. Artifact format
  hints preserve this handoff without exporting original filenames.
- No real engines/models were launched; GPU deployment validation and automatic
  hypothesis loading remain outstanding. See [friend-quickstart.md](friend-quickstart.md).

The public GitHub install command was verified in a fresh environment, followed by
both engine CLI end-to-end protocol tests. No real GPU deployment was launched.
