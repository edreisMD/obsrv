"""Bounded concurrent HTTP scrapes. A failure is recorded, never converted to zero."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .metrics import GPU_METRICS, parse_metrics, serving_metrics

MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def fetch(url, timeout):
    request = Request(url, headers={"Accept": "text/plain", "User-Agent": "serve-observe/0.1"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("Metrics response exceeds 8 MiB limit")
    return body.decode("utf-8")


@dataclass(frozen=True)
class Scrape:
    source: str
    started_ns: int
    ended_ns: int
    wall_time_ns: int
    samples: tuple
    error: str | None = None

    def record(self):
        result = asdict(self)
        result["samples"] = [s.record() for s in self.samples]
        return result


def collect_once(config, *, fetcher=fetch):
    def scrape(source, url, names):
        start = time.monotonic_ns()
        try:
            text = fetcher(url, config.timeout_seconds)
            wall = time.time_ns()
            samples = parse_metrics(text, names, wall / 1e9, config.max_gap_seconds)
            return Scrape(source, start, time.monotonic_ns(), wall, tuple(samples))
        except (HTTPError, URLError, OSError, ValueError) as exc:
            # Avoid persisting endpoint contents, URLs, or exception messages with secrets.
            return Scrape(
                source, start, time.monotonic_ns(), time.time_ns(), (), type(exc).__name__
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(scrape, "gpu", config.dcgm_url, GPU_METRICS),
            pool.submit(scrape, "serving", config.serving_url, serving_metrics(config.engine)),
        ]
        return tuple(f.result() for f in futures)
