# Experiment workflow

Use the collector as a trusted observer and give its exported files to any
optimization agent, notebook or deployment pipeline. No specific optimizer is
required.

1. Pin a replay and record its identity, offered request schedule, model revision,
   precision, hardware, engine version and deployment commit.
2. Warm up the deployment. Start a fresh capture and run the fixed measurement
   workload. Keep model load and warmup separate unless they are the objective.
3. Export the report and samples alongside client benchmark metrics and task
   correctness results. Server latency means do not establish client SLO compliance.
4. Inspect low engine activity during sustained demand. Form one falsifiable
   scheduling or deployment hypothesis; request a focused trace if coarse metrics
   cannot distinguish CPU, communication and GPU work.
5. Change one factor. Repeat at matched load, hardware, cache policy and warmup.
   Alternate baseline/candidate order and collect independent repetitions.
6. Compare correctness, errors, client latency, throughput/cost and evidence
   coverage before promoting a change. Measure collector and trace overhead.

The objective should be goodput or cost per successful task under latency and
correctness constraints. Adding useless GPU work or dropping requests can make
utilization look better while making the service worse.

For agentic workloads, waits between tool calls can be expected. Reducing them
may require routing or multiplexing sessions rather than changing a GPU kernel.
Include bursty arrivals and both low-overlap and high-overlap requests.

## Evidence identity

`config_sha256` identifies the declared capture configuration, not deployment
bytes. Metadata and correctness flags are operator/producer declarations.
Reports and captures are editable local evidence, not an attestation system.
Provide build and benchmark identities separately.

Monotonic clocks cannot be compared across collectors. GPU/MIG and scheduler
identities must not be double counted. This release analyzes one mapped worker
per capture and permits viewing multiple captures together.
