"""Synthetic advertising delivery data with a realistic structure.

The generator exists for one reason: **a detector can only be evaluated when the
ground truth is known**. Real ad logs never tell you which days were anomalous,
so here we simulate a plausible delivery process (keyword pack x placement x
hour block), inject known anomalies, and keep the answer key.

All numbers are synthetic. Nothing here comes from any employer or client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# Mon..Sun demand pattern: weekends are cheaper for B2B-ish intent packs.
WEEKDAY_FACTOR = np.array([1.00, 1.02, 1.03, 1.04, 1.08, 0.92, 0.88], dtype=float)


@dataclass(frozen=True)
class PackSpec:
    """One keyword pack: how much traffic it takes, at what price, converting how well."""

    name: str
    click_share: float
    cpc: float
    cvr: float


PACKS: tuple[PackSpec, ...] = (
    PackSpec("brand", 0.12, 1.80, 0.110),
    PackSpec("competitor", 0.10, 3.60, 0.060),
    PackSpec("high_intent_generic", 0.22, 4.20, 0.070),
    PackSpec("info_generic", 0.26, 2.90, 0.035),
    PackSpec("category_explore", 0.18, 2.40, 0.028),
    PackSpec("long_tail", 0.12, 1.60, 0.020),
)

# placement -> (share of pack traffic, cpc multiplier)
PLACEMENTS: tuple[tuple[str, float, float], ...] = (
    ("search_results", 0.45, 1.12),
    ("search_tabs", 0.30, 0.95),
    ("product_pages", 0.25, 0.85),
)

# hour block -> share of daily traffic
HOUR_BLOCKS: tuple[tuple[str, float], ...] = (
    ("00-08", 0.08),
    ("08-14", 0.34),
    ("14-20", 0.38),
    ("20-24", 0.20),
)


def generate(
    days: int = 180,
    seed: int = 7,
    base_daily_clicks: int = 42_000,
    start: str | date = "2026-01-01",
) -> pd.DataFrame:
    """Return a tidy daily frame at (date, keyword_pack, placement, hour_block) grain.

    Columns: date, keyword_pack, placement, hour_block, clicks, spend, conversions, cpa.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    n = len(dates)

    # --- calendar effects -------------------------------------------------
    weekday = WEEKDAY_FACTOR[dates.dayofweek.to_numpy()]
    annual = 1.0 + 0.06 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.0)
    payday = np.where(np.isin(dates.day.to_numpy(), [9, 10, 11]), 1.04, 1.0)
    calendar = weekday * annual * payday

    # --- daily total clicks ----------------------------------------------
    total_clicks = base_daily_clicks * calendar * rng.lognormal(mean=0.0, sigma=0.05, size=n)

    frames: list[pd.DataFrame] = []
    for spec in PACKS:
        pack_clicks = total_clicks * spec.click_share * rng.lognormal(0.0, 0.04, size=n)
        for placement, share, cpc_mult in PLACEMENTS:
            for block, block_share in HOUR_BLOCKS:
                scale = share * block_share
                clicks = pack_clicks * scale * rng.lognormal(0.0, 0.06, size=n)
                cpc = spec.cpc * cpc_mult * rng.lognormal(0.0, 0.02, size=n)
                spend = clicks * cpc
                # conversion rate drifts a little; search results convert better
                cvr = (
                    spec.cvr
                    * (1.0 + 0.10 * (cpc_mult - 1.0))
                    * rng.lognormal(0.0, 0.03, size=n)
                    * (1.0 + 0.03 * (block_share - 0.25))
                )
                conversions = clicks * np.clip(cvr, 1e-4, 1.0)
                frames.append(
                    pd.DataFrame(
                        {
                            "date": dates,
                            "keyword_pack": spec.name,
                            "placement": placement,
                            "hour_block": block,
                            "clicks": clicks,
                            "spend": spend,
                            "conversions": conversions,
                        }
                    )
                )

    df = pd.concat(frames, ignore_index=True)

    # --- budget cap per pack-day -----------------------------------------
    # A cap that binds occasionally is what makes real series non-Gaussian.
    pack_day = df.groupby(["date", "keyword_pack"])["spend"].transform("median")
    cap_ratio = (1.30 * pack_day / df["spend"]).clip(upper=1.0)
    df["clicks"] = df["clicks"] * cap_ratio
    df["spend"] = df["spend"] * cap_ratio
    df["conversions"] = df["conversions"] * cap_ratio

    df["cpa"] = df["spend"] / df["conversions"].clip(lower=1e-9)
    df = df.sort_values(["date", "keyword_pack", "placement", "hour_block"]).reset_index(drop=True)
    return df


def aggregate(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Aggregate to a coarser grain and recompute the ratio metric (never average a CPA)."""
    keys = [by] if isinstance(by, str) else list(by)
    out = (
        df.groupby(keys, as_index=False)[["clicks", "spend", "conversions"]]
        .sum()
        .sort_values(keys)
    )
    out["cpa"] = out["spend"] / out["conversions"].clip(lower=1e-9)
    return out
