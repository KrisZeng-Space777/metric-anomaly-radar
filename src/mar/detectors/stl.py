"""STL one-step-ahead residual detector.

STL splits a series into trend + seasonal + remainder. Here it is used as a
**forecaster**, not as a smoother:

1. Fit STL on the trailing window that ends *yesterday*.
2. Predict today as last trend + the seasonal value from one period ago.
3. Score the prediction error against the robust spread of **recent forecast
   errors** (not in-sample residuals).

Two details carry the design:

* Fitting on a window that includes today and reading today's residual off the
  edge is quietly broken — the trend component absorbs part of any boundary
  anomaly (low recall) and boundary residuals have a different variance than
  interior ones (bad scale, false alarms).
* In-sample residuals are far too clean to calibrate a threshold: STL's trend and
  seasonal components fit the noise as well, so the robust scale collapses and
  everything looks anomalous. The scale has to come from out-of-sample errors,
  which is what the detector is actually thresholding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .base import as_flags

_MAD_TO_SIGMA = 1.4826


@dataclass
class STLDetector:
    name: str = "stl_residual"
    period: int = 7
    window: int = 84
    min_train: int = 35
    threshold: float = 3.5
    robust: bool = True
    error_buffer: int = 28
    min_errors: int = 14
    _errors: deque = field(default_factory=lambda: deque(maxlen=28), repr=False)

    def detect(self, series: pd.Series) -> pd.Series:
        x = series.astype(float).to_numpy()
        flags = np.zeros(len(x), dtype=bool)
        errors: deque[float] = deque(maxlen=self.error_buffer)

        for t in range(self.min_train, len(x)):
            lo = max(0, t - self.window)
            window = x[lo:t]  # strictly before t
            if len(window) < self.min_train:
                continue
            try:
                res = STL(window, period=self.period, robust=self.robust).fit()
            except Exception:  # pragma: no cover - degenerate windows only
                continue

            forecast = float(res.trend[-1] + res.seasonal[-self.period])
            error = x[t] - forecast

            if len(errors) >= self.min_errors:
                arr = np.asarray(errors, dtype=float)
                scale = _MAD_TO_SIGMA * np.median(np.abs(arr - np.median(arr)))
                if scale > 0 and not np.isnan(scale):
                    flags[t] = abs(error) / scale > self.threshold

            errors.append(error)

        return as_flags(flags, series.index)
