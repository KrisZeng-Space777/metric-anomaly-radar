"""Tabular CUSUM on the seasonally adjusted level.

CUSUM accumulates small deviations instead of reacting to a single large one, so
it is the natural detector for *level shifts* and slow drifts — the two families
a point-wise z-score tends to miss or detect late.

Two implementation details matter:

* The statistic runs on the **level minus a weekday baseline**, not on first
  differences. Differencing turns a step into a single spike followed by zeros,
  which is exactly what CUSUM is bad at accumulating.
* The statistic **resets after an alarm**. Without the reset, one level shift
  alarms every single day for the rest of the series and the false-alarm count
  becomes meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import as_flags

_MAD_TO_SIGMA = 1.4826


@dataclass
class CUSUMDetector:
    name: str = "cusum"
    window: int = 28
    drift: float = 0.5  # allowance k, in robust standard deviations
    threshold: float = 5.0  # decision interval h
    weeks_of_history: int = 4

    def detect(self, series: pd.Series) -> pd.Series:
        x = series.astype(float)
        weekday = x.index.dayofweek
        baseline = x.groupby(weekday).transform(
            lambda col: col.shift(1).rolling(self.weeks_of_history, min_periods=2).median()
        )
        resid = x - baseline

        centre = resid.shift(1).rolling(self.window, min_periods=7).median()
        mad = (resid - centre).abs().shift(1).rolling(self.window, min_periods=7).median()
        scale = _MAD_TO_SIGMA * mad
        z = ((resid - centre) / scale.replace(0.0, np.nan)).to_numpy()

        pos = neg = 0.0
        flags = np.zeros(len(z), dtype=bool)
        for i, zi in enumerate(z):
            if np.isnan(zi):
                pos = neg = 0.0
                continue
            pos = max(0.0, pos + zi - self.drift)
            neg = max(0.0, neg - zi - self.drift)
            if pos > self.threshold or neg > self.threshold:
                flags[i] = True
                pos = neg = 0.0
        return as_flags(flags, series.index)
