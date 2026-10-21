"""Detector interface.

Every detector is a pure function of one univariate daily series: it may only look
at data up to and including day *t* when deciding whether day *t* is anomalous.
That constraint is what makes the benchmark honest — no future information.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, series: pd.Series) -> pd.Series:  # pragma: no cover - protocol
        ...


def as_flags(raw: pd.Series, index: pd.Index) -> pd.Series:
    """Normalise a detector output to a boolean series aligned to ``index``."""
    flags = pd.Series(raw, index=index).fillna(False).astype(bool)
    return flags.rename("is_anomaly")
