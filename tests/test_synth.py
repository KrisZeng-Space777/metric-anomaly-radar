from __future__ import annotations

import numpy as np
import pandas as pd

from mar import synth


def test_grain_is_unique_and_complete():
    df = synth.generate(days=30, seed=1)
    assert df["date"].nunique() == 30
    assert not df.duplicated(["date", "keyword_pack", "placement", "hour_block"]).any()
    expected = 30 * len(synth.PACKS) * len(synth.PLACEMENTS) * len(synth.HOUR_BLOCKS)
    assert len(df) == expected


def test_values_are_positive_and_cpa_consistent():
    df = synth.generate(days=14, seed=2)
    for col in ("clicks", "spend", "conversions"):
        assert (df[col] > 0).all(), col
    assert np.allclose(df["cpa"], df["spend"] / df["conversions"])


def test_same_seed_is_reproducible():
    a = synth.generate(days=10, seed=42)
    b = synth.generate(days=10, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = synth.generate(days=10, seed=1)
    b = synth.generate(days=10, seed=2)
    assert not np.allclose(a["spend"], b["spend"])


def test_weekly_seasonality_is_present():
    df = synth.generate(days=120, seed=3)
    daily = df.groupby("date")["clicks"].sum()
    weekday_means = daily.groupby(daily.index.dayofweek).mean()
    # Friday (4) should clearly out-earn Sunday (6) in this simulated demand pattern
    assert weekday_means.loc[4] > weekday_means.loc[6]


def test_aggregate_recomputes_ratio():
    df = synth.generate(days=5, seed=4)
    agg = synth.aggregate(df, "keyword_pack")
    expected = agg["spend"] / agg["conversions"]
    assert np.allclose(agg["cpa"], expected)
