# Contributing

Bug reports, metric fixtures and improvements to measurement accuracy are welcome.

Use Python 3.11 or newer and `uv sync --frozen --dev`, then run `uv run pytest`,
`uv run ruff check .`, `uv run ruff format --check .`, and `uv build`.
The browser UI has no build step; syntax-check it with
`node --check src/serve_observe/web/app.js`.

Describe the deployment engine/version, metric names and labels, the expected
measurement, and the observed result. Scrub endpoint credentials, deployment
identifiers and private workload details from fixtures before posting them.

Changes to measurements should include a small fixture and a test showing
coverage, missing values and attribution boundaries. Keep unknown values
unknown; never turn a scrape failure into evidence of idle capacity. Dashboard
changes should be checked at desktop and narrow widths.

The project uses the MIT license. Open a GitHub issue to discuss substantial
features before building them.
