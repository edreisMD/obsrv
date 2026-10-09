# Opt-in NVIDIA validation

This release is tested locally with synthetic engine protocols and trace fixtures.
No NVIDIA production measurements or overhead numbers have been established.

Use an existing Linux NVIDIA test deployment before production adoption. Do not run
concurrent GPU benchmark processes or combine different traffic generators in a comparison.

1. Record the image/code digest, model, engine/NVIDIA/CUDA versions and GPU topology.
   Mount one shared trace directory and obsrv state volume per deployment.
2. Configure PyTorch tracing using `obsrv deployment-settings`; deploy through the existing
   deployment process. Enable shape recording. Confirm `/metrics`, model discovery and
   native profiling endpoints are accessible from the collector. For authenticated servers,
   use an environment reference, never a token embedded in the URL/config.
3. Activate obsrv with the deployed revision. Inspect discovered metadata; explicitly map
   real trace filenames to every expected GPU rank before claiming complete worker coverage.
4. Under the same representative coding-agent traffic, measure three separate windows:
   collector disabled, passive collector enabled, and an explicitly profiled window.
   Preserve request-level latency/error data externally for the overhead comparison.
   Report CPU/memory cost of the collector and latency/throughput/error changes; do not
   invent an acceptable overhead threshold before agreeing on the service's SLO.
5. Capture a short window, confirm start/stop/flush behavior, import all worker traces,
   inspect VibeSys certification and verify the exported artifact hashes. Confirm GPU kernels,
   operator shapes and applicable roofline metrics actually exist; CUDA graphs may restrict
   attribution and must be reported rather than disabling production graphs automatically.
6. Repeat for both vLLM and SGLang. Test graceful cancellation, delayed flush, worker absence,
   endpoint errors, a collector restart with an outstanding session, and recovery.
7. Test a controlled revision rollout: update obsrv identity, verify the new cohort stays
   separate, and export both revisions. Different production traffic does not establish a
   causal speedup. Controlled VibeSys experiments remain the acceptance gate.
8. If using Nsight, import an operator-captured `.nsys-rep` or SQLite export and inspect the
   additional CUDA runtime/gap/graph reports. `nsys` is required only for report conversion.

Reference deployment interfaces checked on 2026-10-08:

- https://docs.vllm.ai/en/stable/contributing/profiling/
- https://docs.vllm.ai/en/stable/api/vllm/config/profiler/
- https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx
- https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_components/profiler_manager.py

Engine compatibility is capability-dependent. The current vLLM profiler-config path
requires >=0.13; earlier engines need their supported setup or manual trace import.
SGLang versions must accept record_shapes/activities in start_profile. These HTTP protocols
are fixture-tested here; they are not claimed to have been GPU-tested against installed engines.
