"""Figures for the README. Light theme, English labels, no seaborn dependency."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .anomalies import AnomalyEvent

INK = "#1F2933"
GRID = "#DCE3EA"
TRUTH = "#E8A33D"
MARKERS = {"robust_z": "#2F6FED", "stl_residual": "#0E9F6E", "cusum": "#D64545"}


def plot_example(
    series: pd.Series,
    flags_by_detector: dict[str, pd.Series],
    events: list[AnomalyEvent],
    path: str | Path,
    title: str = "Injected anomaly, detector alarms and ground-truth window",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for ev in events:
        ax.axvspan(pd.Timestamp(ev.start), pd.Timestamp(ev.end), color=TRUTH, alpha=0.16, lw=0)

    ax.plot(series.index, series.to_numpy(), color=INK, lw=1.3, label="observed metric")

    for name, flags in flags_by_detector.items():
        hits = flags[flags]
        ax.scatter(
            hits.index,
            series.reindex(hits.index),
            s=26,
            color=MARKERS.get(name, "#666666"),
            label=f"alarm: {name}",
            zorder=3,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_ylabel("daily spend", color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)

    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_attribution(table: pd.DataFrame, path: str | Path, metric: str = "CPA") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = table.head(6).iloc[::-1]
    colors = ["#D64545" if v > 0 else "#0E9F6E" for v in data["contribution"]]

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.barh(data["member"], data["contribution"], color=colors, height=0.6)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(f"Contribution to the change in {metric} (Shapley)", fontsize=11, loc="left")
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
