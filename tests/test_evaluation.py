from __future__ import annotations

import pandas as pd

from mar.anomalies import AnomalyEvent
from mar.evaluation import PointScore, score_events, score_point


def test_point_score_perfect():
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    flags = pd.Series([False] * 8 + [True, True], index=index)
    truth = flags.copy()
    s = score_point(flags, truth)
    assert (s.tp, s.fp, s.fn) == (2, 0, 0)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0


def test_point_score_counts_false_alarms():
    index = pd.date_range("2026-01-01", periods=10, freq="D")
    flags = pd.Series([True] + [False] * 9, index=index)
    truth = pd.Series([False] * 9 + [True], index=index)
    s = score_point(flags, truth)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)
    assert s.precision == 0.0 and s.recall == 0.0


def test_event_detected_on_the_same_day_has_zero_delay():
    index = pd.date_range("2026-01-01", periods=30, freq="D")
    flags = pd.Series(False, index=index)
    flags.iloc[10] = True
    event = AnomalyEvent("pack:a", "spike", index[10].date(), index[10].date())
    s = score_events(flags, [event], tolerance=3)
    assert s.detected == 1 and s.mttd_days == 0.0


def test_late_alarm_beyond_tolerance_is_a_miss():
    index = pd.date_range("2026-01-01", periods=30, freq="D")
    flags = pd.Series(False, index=index)
    flags.iloc[15] = True
    event = AnomalyEvent("pack:a", "level_shift", index[10].date(), index[12].date())
    s = score_events(flags, [event], tolerance=3)
    assert s.detected == 0
    assert s.false_alarms == 1


def test_false_alarm_rate_excludes_event_windows():
    index = pd.date_range("2026-01-01", periods=30, freq="D")
    flags = pd.Series(False, index=index)
    flags.iloc[10] = True
    flags.iloc[20] = True
    event = AnomalyEvent("pack:a", "spike", index[10].date(), index[10].date())
    s = score_events(flags, [event], tolerance=3)
    assert s.false_alarms == 1
    assert 0 < s.false_alarm_rate < 0.1


def test_point_score_zero_division_safe():
    empty = PointScore(tp=0, fp=0, fn=0)
    assert empty.precision == 0.0 and empty.recall == 0.0 and empty.f1 == 0.0
