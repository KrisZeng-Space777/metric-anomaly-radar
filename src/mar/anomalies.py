"""Anomaly injection with an answer key.

Four anomaly families are injected, each one corresponding to a failure mode that
actually shows up in advertising operations:

- ``level_shift``   : budget rule changed / a big keyword pack paused (step change)
- ``spike``         : tracking outage or double-counting on a single day
- ``trend_change``  : bid landscape drifting over two weeks (gradual)
- ``variance``      : delivery got unstable without moving the mean

Every scenario records which *monitored series* it should be visible in, so the
benchmark can score detectors per series instead of per raw row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    """How to corrupt the data, plus where the anomaly should be detectable."""

    kind: str
    monitored_series: str
    selector: dict[str, str]
    start_offset: int  # days after the warm-up window
    duration: int
    click_curve: Callable[[int], float]
    conv_curve: Callable[[int], float]
    note: str = ""
    scope: str = "series"  # "series" -> only monitored_series is affected; "all" -> every series


@dataclass(frozen=True)
class AnomalyEvent:
    series: str
    kind: str
    start: date
    end: date
    note: str = ""
    scope: str = "series"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.kind} on {self.series} [{self.start} .. {self.end}]"


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------

_WARMUP = 45  # detectors get 45 clean days before anything happens


def _level_shift(i: int) -> float:
    return 1.55


def _spike(i: int) -> float:
    return 2.60 if i == 0 else 1.0


def _ramp(i: int) -> float:
    return 1.0 + 0.5 * i / 13.0


def _flat(i: int) -> float:
    return 1.0


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        kind="level_shift",
        monitored_series="pack:high_intent_generic",
        selector={"keyword_pack": "high_intent_generic"},
        start_offset=8,
        duration=8,
        click_curve=_level_shift,
        conv_curve=lambda i: 1.15,
        note="预算上调，花费跳升而转化跟不上 -> CPA 恶化",
    ),
    Scenario(
        kind="spike",
        monitored_series="total",
        selector={},
        start_offset=32,
        duration=1,
        click_curve=_spike,
        conv_curve=lambda i: 1.20,
        note="全站单日埋点重复上报",
        scope="all",
    ),
    Scenario(
        kind="trend_change",
        monitored_series="pack:competitor",
        selector={"keyword_pack": "competitor"},
        start_offset=52,
        duration=14,
        click_curve=_ramp,
        conv_curve=lambda i: 1.05,
        note="竞品词竞价环境恶化，成本逐步抬升",
    ),
    Scenario(
        kind="variance",
        monitored_series="pack:info_generic",
        selector={"keyword_pack": "info_generic"},
        start_offset=76,
        duration=21,
        click_curve=_flat,
        conv_curve=lambda i: 1.0,
        note="投放稳定性下降，均值不变、方差放大（最难的一类）",
    ),
    Scenario(
        kind="level_shift",
        monitored_series="placement:product_pages",
        selector={"placement": "product_pages"},
        start_offset=104,
        duration=7,
        click_curve=lambda i: 0.55,
        conv_curve=lambda i: 0.60,
        note="版位流量下滑",
    ),
    Scenario(
        kind="spike",
        monitored_series="pack:brand",
        selector={"keyword_pack": "brand"},
        start_offset=128,
        duration=1,
        click_curve=lambda i: 0.35,
        conv_curve=lambda i: 0.40,
        note="品牌词当日掉量（投放中断）",
    ),
)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def inject(
    df: pd.DataFrame,
    scenarios: tuple[Scenario, ...] = SCENARIOS,
    seed: int = 11,
    warmup: int = _WARMUP,
) -> tuple[pd.DataFrame, list[AnomalyEvent]]:
    """Return (corrupted copy of df, list of ground-truth events).

    The corruption is multiplicative on clicks/conversions, so ratios (CPA) move
    the way they do in reality: cost up, conversions lagging behind.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    dates = pd.DatetimeIndex(sorted(out["date"].unique()))
    events: list[AnomalyEvent] = []

    for sc in scenarios:
        start_idx = warmup + sc.start_offset
        end_idx = min(start_idx + sc.duration - 1, len(dates) - 1)
        if start_idx >= len(dates):
            continue
        start, end = dates[start_idx], dates[end_idx]

        mask = (out["date"] >= start) & (out["date"] <= end)
        for column, value in sc.selector.items():
            mask &= out[column] == value

        offset = (out.loc[mask, "date"] - start).dt.days.to_numpy()
        click_mult = np.array([sc.click_curve(int(i)) for i in offset], dtype=float)
        conv_mult = np.array([sc.conv_curve(int(i)) for i in offset], dtype=float)

        if sc.kind == "variance":
            # mean preserving: random log-normal shocks, sigma tripled
            click_mult = click_mult * rng.lognormal(0.0, 0.28, size=mask.sum())

        out.loc[mask, "clicks"] = out.loc[mask, "clicks"].to_numpy() * click_mult
        out.loc[mask, "spend"] = out.loc[mask, "spend"].to_numpy() * click_mult
        out.loc[mask, "conversions"] = out.loc[mask, "conversions"].to_numpy() * conv_mult

        events.append(
            AnomalyEvent(
                series=sc.monitored_series,
                kind=sc.kind,
                start=start.date(),
                end=end.date(),
                note=sc.note,
                scope=sc.scope,
            )
        )

    out["cpa"] = out["spend"] / out["conversions"].clip(lower=1e-9)
    return out, events


# ---------------------------------------------------------------------------
# Monitored series
# ---------------------------------------------------------------------------

MONITORED_METRIC = "spend"


def monitored_series(df: pd.DataFrame, metric: str = MONITORED_METRIC) -> dict[str, pd.Series]:
    """Build the univariate daily series each detector is scored on.

    Keys follow the ground-truth naming: ``total``, ``pack:<name>``, ``placement:<name>``.
    Series are reindexed to a complete daily calendar so gaps are visible, not skipped.
    """
    calendar = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    series: dict[str, pd.Series] = {}

    series["total"] = (
        df.groupby("date")[metric].sum().reindex(calendar).astype(float)
    )
    for pack, grp in df.groupby("keyword_pack"):
        series[f"pack:{pack}"] = (
            grp.groupby("date")[metric].sum().reindex(calendar).astype(float)
        )
    for placement, grp in df.groupby("placement"):
        series[f"placement:{placement}"] = (
            grp.groupby("date")[metric].sum().reindex(calendar).astype(float)
        )
    return {k: v.interpolate(limit_direction="both") for k, v in series.items()}


def events_to_dicts(events: list[AnomalyEvent]) -> list[dict[str, str]]:
    """Serialise the answer key so a benchmark run can be reproduced from disk."""
    return [
        {
            "series": e.series,
            "kind": e.kind,
            "start": e.start.isoformat(),
            "end": e.end.isoformat(),
            "note": e.note,
            "scope": e.scope,
        }
        for e in events
    ]


def events_from_dicts(rows: list[dict[str, str]]) -> list[AnomalyEvent]:
    return [
        AnomalyEvent(
            series=r["series"],
            kind=r["kind"],
            start=date.fromisoformat(r["start"]),
            end=date.fromisoformat(r["end"]),
            note=r.get("note", ""),
            scope=r.get("scope", "series"),
        )
        for r in rows
    ]


def truth_frame(
    events: list[AnomalyEvent],
    calendar: pd.DatetimeIndex,
    series_keys: list[str] | None = None,
) -> dict[str, pd.Series]:
    """Explode events into boolean per-series truth vectors (for point-wise scoring).

    An event with ``scope="all"`` (e.g. a tracking outage that hits every account
    at once) is legitimately visible in every series, so marking only the ``total``
    series would score correct alarms on other series as false positives.
    """
    truth: dict[str, pd.Series] = {}
    for ev in events:
        flag = pd.Series(False, index=calendar)
        flag.loc[(calendar >= pd.Timestamp(ev.start)) & (calendar <= pd.Timestamp(ev.end))] = True
        targets = [ev.series] if ev.scope == "series" else list(series_keys or [ev.series])
        for key in targets:
            truth[key] = flag if key not in truth else (truth[key] | flag)
    return truth
