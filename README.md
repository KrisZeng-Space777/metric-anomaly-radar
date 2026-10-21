# metric-anomaly-radar

Anomaly detection and dimension-level attribution for advertising metrics.

Project skeleton: packaging, lint/test configuration, and the directory layout.
The data generator, detectors and evaluation protocol land in the following commits.

## Layout

```
data/      generated datasets (git-ignored)
results/   benchmark output and figures
src/mar/   package
scripts/   one-off analysis helpers
tests/     pytest suite
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
