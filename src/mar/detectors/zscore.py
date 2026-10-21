"""Rolling robust z-score.

Why MAD instead of mean/std: advertising series contain the very spikes we are
trying to catch, and a standard deviation computed over such a window inflates
until nothing looks anomalous anymore (masking effect).

Why a weekday baseline: ad demand has a strong weekly cycle. Subtracting a
trailing weekday median first removes it without fitting a full seasonal model,
which keeps the detector cheap enough to run on hundreds of series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import as_flags

_MAD_TO_SIGMA = 1.4826


@dataclass
class RobustZDetector:
    name: str = "robust_z"
    window: int = 28
    threshold: float = 4.0
    seasonal: str = "weekday"  # {"weekday", "none"}
    weeks_of_history: int = 4

    def residuals(self, series: pd.Series) -> pd.Series:
        x = series.astype(float)
        if self.seasonal == "weekday":
            weekday = x.index.dayofweek
            # trailing median of the same weekday over the previous N weeks
            baseline = x.groupby(weekday).transform(
                lambda col: col.shift(1).rolling(self.weeks_of_history, min_periods=2).median()
            )
            return x - baseline
        baseline = x.shift(1).rolling(self.window, min_periods=7).median()
        return x - baseline

    def detect(self, series: pd.Series) -> pd.Series:
        resid = self.residuals(series)
        centre = resid.shift(1).rolling(self.window, min_periods=7).median()
        mad = (resid - centre).abs().shift(1).rolling(self.window, min_periods=7).median()
        scale = (_MAD_TO_SIGMA * mad).replace(0.0, np.nan)
        z = (resid - centre) / scale
        return as_flags(z.abs() > self.threshold, series.index)
