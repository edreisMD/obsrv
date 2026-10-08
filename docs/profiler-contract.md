# Measurement provenance and parity

Vendored from `uw-syfi/vibesys` revision
`41dd0a8eb2db160c3ae43e4472401cd663f839e5`:

- `resources/profilers/torch/analyze_torch_profile.py`
- `resources/profilers/nsys/analyze_nsys.py`

Source and vendored byte hashes are in `src/obsrv/_vendor/provenance.json`.
The MIT license is preserved beside the vendored code. Source hashes record the
actual files copied from the local checkout, so provenance remains explicit even
if the source checkout has uncommitted changes.

Measurement algorithms and human-readable analysis commands are retained. Packaging
changes disable the standalone main entrypoints and eager PyTorch import, and replace
Nsight's capture-runtime exception dependency with a small local analysis exception.
Obsrv does not expose the upstream model-loading, synthetic load or injection capture
functions. Capture is implemented separately through native serving control endpoints.

Torch exports preserve the upstream summary, trace certification, correlated GEMM
shapes and available roofline rows, plus exact summary/shape/roofline report text. Missing
shapes, correlations, peaks or CUDA graph annotations limit attribution; obsrv does not
synthesize missing operator measurements. A certification PASS is trace validity, not
model correctness or an efficiency win.

Nsight exports preserve kernel, CPU overhead, idle gap, memory, graph replay and inferred
step timeline reports. The upstream kernel report's percentages and displayed total use
its selected top-N rows. Obsrv separately reports all-kernel count/duration sums and
per-device union/gap measurements. The upstream idle-gap metric excludes gaps <=1us
and trace edges. Graph replay/step attribution require the corresponding captured tables.

Nsight Compute is a separate VibeSys kernel-writing profiler, not the default LLM-serving
profiler. ROCm, Trainium, CPU and OpenTelemetry profilers target other backends/domains.
They are not part of this NVIDIA vLLM/SGLang package release. DCGM and serving Prometheus
sources are supplementary, not replacements for these VibeSys profilers.

Local tests compare PyTorch numerical outputs, certification and reports against the
original VibeSys source, and compare Nsight overlap calculations/reports with its analyzer.
Fixtures intentionally contain overlapping kernels, incomplete captures and missing tables.
