"""Hyper-parameter sweep on a *development* seed.

The benchmark in the README is reported on ``seed=7``. Every threshold used there
was chosen here, on ``seed=1`` — a different draw of the same generator. Keeping
the two apart is the cheapest way to avoid reporting numbers that are really just
the result of tuning on the test set.

    python scripts/tune_detectors.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mar import anomalies, benchmark, synth  # noqa: E402
from mar.detectors import CUSUMDetector, RobustZDetector, STLDetector  # noqa: E402

DEV_SEED = 1


def main() -> None:
    clean = synth.generate(days=180, seed=DEV_SEED)
    dirty, events = anomalies.inject(clean, seed=101)

    rows = []

    for window, threshold in itertools.product((21, 28, 42), (3.0, 3.5, 4.0)):
        factory = lambda w=window, t=threshold: RobustZDetector(window=w, threshold=t)  # noqa: E731
        row = benchmark.run(dirty, events, detectors=(factory,)).table.iloc[0]
        rows.append(_row("robust_z", window, threshold, row))

    for window, threshold in itertools.product((56, 84, 112), (3.0, 3.5, 4.0, 4.5)):
        factory = lambda w=window, t=threshold: STLDetector(window=w, threshold=t)  # noqa: E731
        row = benchmark.run(dirty, events, detectors=(factory,)).table.iloc[0]
        rows.append(_row("stl_residual", window, threshold, row))

    for threshold in (4.0, 5.0, 6.0):
        factory = lambda t=threshold: CUSUMDetector(threshold=t)  # noqa: E731
        row = benchmark.run(dirty, events, detectors=(factory,)).table.iloc[0]
        rows.append(_row("cusum", 28, threshold, row))

    table = pd.DataFrame(rows).sort_values(["detector", "f1"], ascending=[True, False])
    print(f"development seed = {DEV_SEED} (benchmark in README uses seed = 7)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _row(name: str, p1: float, p2: float, row: pd.Series) -> dict[str, object]:
    return {
        "detector": name,
        "param1": p1,
        "param2": p2,
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "event_recall": row["event_recall"],
        "mttd_days": row["mttd_days"],
    }


if __name__ == "__main__":
    main()
