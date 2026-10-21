# metric-anomaly-radar

**Anomaly detection and dimension-level attribution for advertising metrics — built so the detector can be evaluated, not just demoed.**

[![ci](https://github.com/KrisZeng-Space777/metric-anomaly-radar/actions/workflows/ci.yml/badge.svg)](https://github.com/KrisZeng-Space777/metric-anomaly-radar/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Problem

A daily delivery report tells you that CPA went up 12% yesterday. It does not tell you whether that is a real anomaly or the weekly cycle, and it does not tell you which slice moved. Two things are missing:

1. **A detector you can trust.** Precision and recall of a detector are unknowable unless the ground truth is known — and real ad logs never carry a label saying "tracking broke on 12 May".
2. **Attribution that reconstructs the number.** CPA is a ratio. Segment deltas do not add up to the total change, so the "which keyword pack caused it" answer quietly does not sum.

```bash
mar demo      # ~10s: synthetic data -> detection -> attribution -> figures
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
mar demo
pytest
```

## What it does

- **Synthetic delivery generator** at `(date, keyword pack, placement, hour block)` grain: weekly cycle, annual drift, payday bump, per-pack CPC/CVR, occasional budget cap. Seeded and documented, so results are reproducible.
- **Answer key**: four anomaly families injected with exact windows — level shift, spike, trend change, variance — each tagged with its scope so a site-wide outage counts as one incident.
- **Three detectors** under one leakage-free protocol: rolling robust z-score (weekday median + MAD), one-step-ahead STL residuals, tabular CUSUM.
- **Two-level scoring**: point-wise precision/recall plus event-wise recall @ 3 days, mean time to detect, and false alarms per 100 series-days.
- **Shapley attribution** of the change in a ratio metric, which sums to the total by construction.

## Layout

```
src/mar/
  synth.py         data generating process
  anomalies.py     injection + answer key + monitored series
  detectors/       robust_z, stl_residual, cusum
  evaluation.py    point-wise and event-wise scoring
  attribution.py   exact Shapley on metric change
  benchmark.py     orchestration + markdown rendering
  plotting.py      figures
  cli.py           mar demo / synth / benchmark / attribute
tests/             30 tests
```

## Roadmap

- [x] data generator, injection, three detectors, scoring, Shapley attribution
- [ ] hyper-parameter sweep on a separate development seed
- [ ] figures + full results section
- [ ] method notes and limitations

## License

MIT
