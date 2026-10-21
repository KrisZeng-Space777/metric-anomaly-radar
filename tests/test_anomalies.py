from __future__ import annotations

import pandas as pd
import pytest

from mar import anomalies, synth


@pytest.fixture(scope="module")
def injected():
    clean = synth.generate(days=180, seed=7)
    return clean, *anomalies.inject(clean)


def test_injection_keeps_schema_and_positivity(injected):
    clean, dirty, _ = injected
    assert list(clean.columns) == list(dirty.columns)
    assert len(clean) == len(dirty)
    for col in ("clicks", "spend", "conversions"):
        assert (dirty[col] > 0).all()


def test_events_are_recorded_with_valid_windows(injected):
    _, _, events = injected
    assert len(events) == len(anomalies.SCENARIOS)
    for ev in events:
        assert ev.start <= ev.end
        assert ev.kind in {"level_shift", "spike", "trend_change", "variance"}


def test_level_shift_actually_moves_the_monitored_series(injected):
    clean, dirty, events = injected
    target = "pack:high_intent_generic"
    before = anomalies.monitored_series(clean)[target]
    after = anomalies.monitored_series(dirty)[target]

    event = next(e for e in events if e.series == target)
    inside = (after.index >= pd.Timestamp(event.start)) & (after.index <= pd.Timestamp(event.end))
    outside = (after.index < pd.Timestamp(event.start) - pd.Timedelta(days=7)) | (
        after.index > pd.Timestamp(event.end) + pd.Timedelta(days=7)
    )
    assert after[inside].mean() > 1.3 * before[outside].mean()


def test_events_round_trip_through_json(injected):
    _, _, events = injected
    rows = anomalies.events_to_dicts(events)
    back = anomalies.events_from_dicts(rows)
    assert back == events


def test_truth_frame_marks_exactly_the_event_window(injected):
    _, _, events = injected
    calendar = pd.date_range("2026-01-01", periods=180, freq="D")
    truth = anomalies.truth_frame(events, calendar)
    event = events[0]
    series = truth[event.series]
    assert series.loc[pd.Timestamp(event.start) : pd.Timestamp(event.end)].all()
    assert not series.loc[pd.Timestamp(event.end) + pd.Timedelta(days=2)]
