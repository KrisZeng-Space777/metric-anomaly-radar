"""Benchmark runner: one table, three detectors, ten monitored series."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from .anomalies import AnomalyEvent, monitored_series, truth_frame
from .detectors import ALL_DETECTORS, Detector
from .evaluation import evaluate


@dataclass
class BenchmarkResult:
    table: pd.DataFrame
    flags: dict[str, dict[str, pd.Series]]


def run(
    df: pd.DataFrame,
    events: list[AnomalyEvent],
    detectors: tuple[type[Detector], ...] | None = None,
    tolerance: int = 3,
    metric: str = "spend",
) -> BenchmarkResult:
    """Score every detector on every monitored series, with wall-clock time."""
    detectors = detectors or ALL_DETECTORS
    series = monitored_series(df, metric=metric)
    calendar = pd.DatetimeIndex(next(iter(series.values())).index)
    truth = truth_frame(events, calendar, series_keys=list(series))

    rows = []
    all_flags: dict[str, dict[str, pd.Series]] = {}

    for factory in detectors:
        detector = factory()
        started = time.perf_counter()
        flags = {key: detector.detect(s) for key, s in series.items()}
        elapsed = time.perf_counter() - started
        all_flags[detector.name] = flags

        scores = evaluate(flags, truth, events, tolerance=tolerance)
        scores["detector"] = detector.name
        scores["seconds"] = round(elapsed, 2)
        rows.append(scores)

    table = pd.DataFrame(rows)[
        [
            "detector",
            "precision",
            "recall",
            "f1",
            "event_recall",
            "events_detected",
            "mttd_days",
            "false_alarms_per_100_series_days",
            "seconds",
        ]
    ].sort_values("f1", ascending=False)

    return BenchmarkResult(table=table.reset_index(drop=True), flags=all_flags)


def to_markdown(table: pd.DataFrame) -> str:
    """Minimal markdown renderer so README tables are reproducible without extra deps."""
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
