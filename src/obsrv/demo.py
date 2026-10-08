"""Deterministic synthetic metric captures. No claimed hardware measurements."""

import json

from .analysis import analyze
from .collector import Scrape
from .config import GPU, Config
from .metrics import GPU_METRICS, parse_metrics, serving_metrics
from .store import Store


def demo_frames(config):
    frames = []
    # Stable no-demand, demand stalls, busy work, outage and recovery.
    activity = [0, 0, 0, 0.02, 0.02, 0.9, 0.9, None, 0.8, 0.8]
    demands = [0, 0, 0, 4, 4, 4, 4, 4, 4, 4]
    for i, (active, demand) in enumerate(zip(activity, demands, strict=True)):
        start = (i + 1) * 1_000_000_000
        gpu_text = f'DCGM_FI_PROF_GR_ENGINE_ACTIVE{{UUID="GPU-DEMO"}} {active or 0}\n'
        gpu_text += f'DCGM_FI_PROF_SM_ACTIVE{{UUID="GPU-DEMO"}} {0.2 if demand else 0}\n'
        gpu_samples = tuple(parse_metrics(gpu_text, GPU_METRICS, i + 1, 3))
        serving_text = (
            f'{config.engine}:num_requests_running{{model_name="demo"}} {demand}\n'
            f'{config.engine}:num_requests_waiting{{model_name="demo"}} {demand}\n'
            f'{config.engine}:generation_tokens_total{{model_name="demo"}} {i * 30}\n'
            f'{config.engine}:time_to_first_token_seconds_sum{{model_name="demo"}} {i * 0.3}\n'
            f'{config.engine}:time_to_first_token_seconds_count{{model_name="demo"}} {i}\n'
        )
        serving_samples = tuple(
            parse_metrics(serving_text, serving_metrics(config.engine), i + 1, 3)
        )
        frames.append(
            (
                Scrape(
                    "gpu",
                    start,
                    start + 1_000_000,
                    start,
                    gpu_samples if active is not None else (),
                    "SyntheticOutage" if active is None else None,
                ),
                Scrape("serving", start, start + 1_000_000, start, serving_samples),
            )
        )
    return frames


def make_demo(directory):
    directory.mkdir(parents=True, exist_ok=True)
    config = Config(
        "synthetic-worker",
        "vllm",
        "http://localhost:8000/metrics",
        "http://localhost:9400/metrics",
        (GPU("demo-gpu", {"UUID": "GPU-DEMO"}),),
        serving_labels={"model_name": "demo"},
        allocation="exclusive",
        metadata={"data_kind": "synthetic", "candidate_revision": "demo-only"},
    )
    with Store(directory / "capture.sqlite") as store:
        run = store.start(config)
        frames = demo_frames(config)
        for seq, frame in enumerate(frames):
            store.append(run, seq, frame)
        store.finish(run)
        store.export_jsonl(directory / "samples.jsonl", run)
        report = {**analyze(config, frames), "run_id": run, "status": "complete"}
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"data_kind": "synthetic", "run_id": run, "report": str(directory / "report.json")}
