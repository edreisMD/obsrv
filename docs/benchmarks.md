# Paired benchmark files

Place UTF-8 JSON files in a directory and pass `--benchmark-dir DIRECTORY` to
the dashboard. Files are read, never executed or edited. Non-benchmark JSON is
ignored. Symlinks and files over 4 MiB are skipped. Invalid benchmark files are
reported as source issues.

```json
{
  "mode": "benchmark",
  "passed": true,
  "scope": "Fixed code-edit replay, one request at a time",
  "workload_sha256": "YOUR-REPLAY-HASH",
  "candidate_sha256": "YOUR-CANDIDATE-HASH",
  "model_revision": "YOUR-MODEL-REVISION",
  "device": {"device_name": "YOUR-GPU"},
  "records": [
    {"case": "edit-1", "reference_s": 2.4, "candidate_s": 1.8, "output_tokens": 120},
    {"case": "edit-2", "reference_s": 2.6, "candidate_s": 2.0, "output_tokens": 120}
  ]
}
```

Numbers above illustrate the schema; they are not measured results. Add
`"data_kind": "synthetic"` to label a demonstration. If token counts include
EOS, use `output_tokens_including_eos` consistently. Optional token counts can be
omitted; throughput then remains unknown.

`passed` is the producer's correctness claim, not independent verification.
Run your fixed output/task checks before setting it. The dashboard keeps
unverified timings visible and does not mark them accepted or deploy changes.

- Speedup: geometric mean of each `reference_s / candidate_s`.
- Total latency reduction: `1 - sum(candidate_s) / sum(reference_s)`.
- Display latency: median for each side.
- Throughput: total declared output tokens divided by total inference time.
- History order: file modification time, not an attested experiment timestamp.

Each artifact is an internally paired experiment. History does not establish
comparability between different models, loads or hardware; nor does it show
cumulative improvements. Repetitions within an artifact are distinct from
independent repeated experiments. GPU inactivity is unavailable in timing-only
artifacts; collect DCGM separately for that evidence.
