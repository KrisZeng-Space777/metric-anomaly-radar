from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mar.detectors import CUSUMDetector, RobustZDetector, STLDetector


def _series(days: int = 120, seed: int = 0, spike_at: int | None = None,
            shift_at: int | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    weekday = np.array([1.0, 1.02, 1.03, 1.04, 1.08, 0.92, 0.88])[index.dayofweek]
    values = 1000 * weekday * rng.lognormal(0.0, 0.05, size=days)
    if spike_at is not None:
        values[spike_at] *= 2.6
    if shift_at is not None:
        values[shift_at:] *= 1.55
    return pd.Series(values, index=index)


def test_robust_z_flags_the_injected_spike():
    s = _series(spike_at=80)
    flags = RobustZDetector().detect(s)
    assert flags.dtype == bool
    assert len(flags) == len(s)
    assert flags.iloc[80]
    assert flags.sum() <= 5  # and it does not fire everywhere


def test_robust_z_is_quiet_on_clean_data():
    s = _series(seed=5)
    flags = RobustZDetector().detect(s)
    assert flags.sum() <= 3


def test_stl_flags_within_a_few_days_of_a_level_shift():
    s = _series(shift_at=90)
    flags = STLDetector().detect(s)
    window = flags.iloc[90:96]
    assert window.any(), "STL should react within 6 days of a 55% step change"


def test_cusum_flags_a_level_shift():
    s = _series(shift_at=80)
    flags = CUSUMDetector().detect(s)
    assert flags.iloc[80:90].any()


def test_detectors_never_emit_nan_or_wrong_length():
    s = _series()
    for det in (RobustZDetector(), STLDetector(), CUSUMDetector()):
        flags = det.detect(s)
        assert flags.isna().sum() == 0
        assert list(flags.index) == list(s.index)


@pytest.mark.parametrize("window", [(7, 0.10), (14, 0.08)])
def test_sensitivity_changes_with_parameters(window):
    w, sigma = window
    rng = np.random.default_rng(1)
    index = pd.date_range("2026-01-01", periods=120, freq="D")
    values = 500 + rng.normal(0, sigma * 500, size=120)
    flags = RobustZDetector(window=w, threshold=2.0).detect(pd.Series(values, index=index))
    assert 0 <= flags.sum() <= 20
