from __future__ import annotations

import pandas as pd
import pytest

from mar import attribution


def _frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["keyword_pack", "spend", "conversions"]).assign(
        clicks=lambda d: d["spend"] / 2.0
    )


def test_additive_metric_contributions_sum_to_the_change():
    base = _frame([("a", 100, 10), ("b", 100, 10), ("c", 50, 5)])
    cur = _frame([("a", 200, 10), ("b", 100, 10), ("c", 50, 5)])
    table = attribution.shapley_change(base, cur, by="keyword_pack", metric="spend")
    assert table["contribution"].sum() == pytest.approx(100.0)
    assert table.iloc[0]["member"] == "a"


def test_ratio_metric_contributions_sum_to_the_change():
    base = _frame([("a", 100, 10), ("b", 100, 10)])
    cur = _frame([("a", 200, 10), ("b", 100, 10)])
    table = attribution.shapley_change(base, cur, by="keyword_pack", metric="cpa")

    base_cpa = 200 / 20
    cur_cpa = 300 / 20
    assert table["contribution"].sum() == pytest.approx(cur_cpa - base_cpa)
    assert table.iloc[0]["member"] == "a"


def test_proportional_growth_contributes_nothing():
    """If every member scales volume by the same factor, the ratio is unchanged.

    This is the property a naive "sum of segment deltas" split gets wrong: the
    ratio metric is scale invariant, so contributions must vanish.
    """
    base = _frame([("a", 100, 10), ("b", 100, 10)])
    cur = _frame([("a", 200, 20), ("b", 300, 30)])
    table = attribution.shapley_change(base, cur, by="keyword_pack", metric="cpa")
    assert table["contribution"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert table["contribution"].sum() == pytest.approx(0.0, abs=1e-9)


def test_missing_member_is_treated_as_zero_volume():
    base = _frame([("a", 100, 10)])
    cur = _frame([("a", 100, 10), ("new", 40, 2)])
    table = attribution.shapley_change(base, cur, by="keyword_pack", metric="spend")
    assert set(table["member"]) == {"a", "new"}
    assert table["contribution"].sum() == pytest.approx(40.0)


def test_explain_is_human_readable():
    base = _frame([("a", 100, 10), ("b", 100, 10)])
    cur = _frame([("a", 300, 10), ("b", 100, 10)])
    text = attribution.explain(base, cur, by="keyword_pack", metric="cpa")
    assert "cpa" in text and "keyword_pack" in text and "a" in text


def test_empty_frames_return_empty_table():
    empty = _frame([])
    table = attribution.shapley_change(empty, empty, by="keyword_pack", metric="cpa")
    assert table.empty
