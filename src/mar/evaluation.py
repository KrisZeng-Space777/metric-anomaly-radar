"""Scoring.

Two levels, because they answer different questions:

* **Point-wise** precision / recall — how many flagged days were real. Cheap, but a
  detector can score well by firing on the first day of every event and never again.
* **Event-wise** recall @ tolerance + mean time to detect (MTTD) — did the alert
  arrive at all, and how late. This is what an on-call engineer actually cares about.

Reporting only one of the two hides the trade-off, so both are always reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomalies import AnomalyEvent


@dataclass
class PointScore:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class EventScore:
    detected: int
    total: int
    mttd_days: float
    false_alarms: int
    clean_days: int

    @property
    def recall(self) -> float:
        return self.detected / self.total if self.total else 0.0

    @property
    def false_alarm_rate(self) -> float:
        """False alarms per monitored series-day."""
        return self.false_alarms / self.clean_days if self.clean_days else 0.0


def _window(index: pd.DatetimeIndex, event: AnomalyEvent) -> np.ndarray:
    return (index >= pd.Timestamp(event.start)) & (index <= pd.Timestamp(event.end))


def score_point(flags: pd.Series, truth: pd.Series) -> PointScore:
    f = flags.reindex(truth.index).fillna(False).to_numpy(dtype=bool)
    t = truth.reindex(flags.index).fillna(False).to_numpy(dtype=bool)
    return PointScore(
        tp=int(np.sum(f & t)),
        fp=int(np.sum(f & ~t)),
        fn=int(np.sum(~f & t)),
    )


def score_events(flags: pd.Series, events: list[AnomalyEvent], tolerance: int = 3) -> EventScore:
    """Single-series event scoring (kept for direct use and tests)."""
    index = pd.DatetimeIndex(flags.index)
    detected, delays, covered = 0, [], np.zeros(len(index), dtype=bool)

    for ev in events:
        lo = pd.Timestamp(ev.start) - pd.Timedelta(days=1)
        hi = pd.Timestamp(ev.start) + pd.Timedelta(days=tolerance)
        window = flags.loc[(index >= lo) & (index <= hi)]
        if window.any():
            detected += 1
            first = window[window].index[0]
            delays.append(max(0, (first - pd.Timestamp(ev.start)).days))
        covered |= (index >= lo) & (index <= pd.Timestamp(ev.end) + pd.Timedelta(days=1))

    return EventScore(
        detected=detected,
        total=len(events),
        mttd_days=float(np.mean(delays)) if delays else float("nan"),
        false_alarms=int(np.sum(flags.to_numpy(dtype=bool) & ~covered)),
        clean_days=int(np.sum(~covered)),
    )


def evaluate(
    flags_by_series: dict[str, pd.Series],
    truth_by_series: dict[str, pd.Series],
    events: list[AnomalyEvent],
    tolerance: int = 3,
) -> dict[str, float]:
    """Aggregate scores across every monitored series for one detector.

    Events with ``scope="all"`` are counted once and are detectable in any series,
    because one tracking outage firing ten series is one incident, not ten.
    """
    tp = fp = fn = 0
    for key, flags in flags_by_series.items():
        if key not in truth_by_series:
            continue
        ps = score_point(flags, truth_by_series[key])
        tp, fp, fn = tp + ps.tp, fp + ps.fp, fn + ps.fn

    index = pd.DatetimeIndex(next(iter(flags_by_series.values())).index)
    covered = {key: np.zeros(len(index), dtype=bool) for key in flags_by_series}
    detected, delays = 0, []

    for ev in events:
        targets = [ev.series] if ev.scope == "series" else list(flags_by_series)
        lo = pd.Timestamp(ev.start) - pd.Timedelta(days=1)
        hi = pd.Timestamp(ev.start) + pd.Timedelta(days=tolerance)

        delay: int | None = None
        for key in targets:
            if key not in flags_by_series:
                continue
            flags = flags_by_series[key]
            window = flags.loc[(index >= lo) & (index <= hi)]
            if window.any():
                first = window[window].index[0]
                d = max(0, (first - pd.Timestamp(ev.start)).days)
                delay = d if delay is None else min(delay, d)
        if delay is not None:
            detected += 1
            delays.append(delay)

        mask = (index >= lo) & (index <= pd.Timestamp(ev.end) + pd.Timedelta(days=1))
        for key in targets:
            if key in covered:
                covered[key] |= mask

    false_alarms = sum(
        int(np.sum(flags.to_numpy(dtype=bool) & ~covered[key])) for key, flags in flags_by_series.items()
    )
    clean_days = sum(int(np.sum(~covered[key])) for key in flags_by_series)
    point = PointScore(tp=tp, fp=fp, fn=fn)

    return {
        "precision": round(point.precision, 3),
        "recall": round(point.recall, 3),
        "f1": round(point.f1, 3),
        "event_recall": round(detected / len(events), 3) if events else 0.0,
        "events_detected": f"{detected}/{len(events)}",
        "mttd_days": round(float(np.mean(delays)), 2) if delays else float("nan"),
        "false_alarms_per_100_series_days": round(100 * false_alarms / clean_days, 2)
        if clean_days
        else 0.0,
    }
