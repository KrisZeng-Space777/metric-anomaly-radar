"""Dimension-level attribution: which slice explains the change?

The naive answer — "compare each segment's average, add them up" — breaks for
ratio metrics such as CPA = spend / conversions, because ratios do not add up.
A segment can have a worse CPA and still contribute nothing to the total change
if it barely has volume.

We therefore treat each dimension member as a player in a cooperative game and
compute its Shapley value on the *change* of the metric:

    v(S) = metric_current(S) - metric_baseline(S),   v(empty) = 0

The Shapley values always sum to the total change v(all) and correctly handle
interaction between segments (the reason this beats a naive additive split when
the metric is a ratio). With <= 10 members the exact computation is trivial
(2^10 subsets); beyond that the caller should sample permutations instead.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

ADDITIVE_METRICS = {"clicks", "spend", "conversions", "impressions"}


def _totals(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    out = frame.groupby(by, as_index=False)[["clicks", "spend", "conversions"]].sum()
    return out.set_index(by)


def _metric_value(row: pd.Series, metric: str) -> float:
    if metric in ADDITIVE_METRICS:
        return float(row[metric])
    if metric == "cpa":
        return float(row["spend"] / row["conversions"]) if row["conversions"] > 0 else 0.0
    raise ValueError(f"unsupported metric: {metric}")


def _value(subset: frozenset[str], cur: pd.DataFrame, base: pd.DataFrame, metric: str) -> float:
    if not subset:
        return 0.0
    idx = list(subset)
    cur_s = cur.reindex(idx).fillna(0.0)
    base_s = base.reindex(idx).fillna(0.0)
    if metric in ADDITIVE_METRICS:
        return float(cur_s[metric].sum() - base_s[metric].sum())
    cur_den = cur_s["conversions"].sum()
    base_den = base_s["conversions"].sum()
    cur_num = cur_s["spend"].sum()
    base_num = base_s["spend"].sum()
    cur_ratio = cur_num / cur_den if cur_den > 0 else 0.0
    base_ratio = base_num / base_den if base_den > 0 else 0.0
    return float(cur_ratio - base_ratio)


def shapley_change(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    by: str,
    metric: str = "cpa",
) -> pd.DataFrame:
    """Return each member's contribution to the change in ``metric``.

    Parameters
    ----------
    baseline, current : tidy frames at least as granular as ``by``.
    by : dimension column, e.g. ``"keyword_pack"`` or ``"hour_block"``.
    metric : one of ``cpa``, ``spend``, ``clicks``, ``conversions``.

    Returns
    -------
    DataFrame with columns ``member``, ``contribution``, ``share``, sorted by the
    absolute contribution. ``contribution`` sums to the total change.
    """
    base = _totals(baseline, by)
    cur = _totals(current, by)
    members = sorted(set(base.index) | set(cur.index))
    n = len(members)
    if n == 0:
        return pd.DataFrame(columns=["member", "contribution", "share"])
    if n > 10:
        raise ValueError("exact Shapley is limited to 10 members; sample permutations above that")

    values: dict[frozenset[str], float] = {}
    for size in range(n + 1):
        for combo in combinations(members, size):
            key = frozenset(combo)
            values[key] = _value(key, cur, base, metric)

    phi: dict[str, float] = {}
    for member in members:
        total = 0.0
        others = [m for m in members if m != member]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in combinations(others, size):
                s = frozenset(combo)
                total += weight * (values[s | {member}] - values[s])
        phi[member] = total

    total_change = values[frozenset(members)]
    out = pd.DataFrame({"member": members, "contribution": [phi[m] for m in members]})
    out["share"] = out["contribution"] / total_change if total_change else np.nan
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def top_contributors(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    by: str,
    metric: str = "cpa",
    k: int = 3,
) -> pd.DataFrame:
    """The ``k`` members responsible for most of the change (same sign as total)."""
    table = shapley_change(baseline, current, by=by, metric=metric)
    if table.empty:
        return table
    return table.head(k)


def explain(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    by: str,
    metric: str = "cpa",
    k: int = 3,
) -> str:
    """One-line human summary, the kind you paste into an alert message."""
    table = shapley_change(baseline, current, by=by, metric=metric)
    if table.empty:
        return "no data"
    total = table["contribution"].sum()
    head = table.head(k)
    parts = [
        f"{row.member} {row.contribution:+.3g} ({row.share:+.0%})"
        for row in head.itertuples()
        if abs(row.contribution) > 0
    ]
    absolute = table["contribution"].abs().sum()
    covered = head["contribution"].abs().sum() / absolute if absolute else 0.0
    return (
        f"{metric} changed {total:+.3g}; the top {k} {by} slices carry {covered:.0%} "
        "of the total absolute attribution: " + ", ".join(parts)
    )
