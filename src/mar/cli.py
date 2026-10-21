"""Command line entry point: ``mar``.

    mar demo                 # synth -> inject -> benchmark -> figures -> attribution
    mar synth  --out data    # write a clean dataset
    mar benchmark --data data/ad_metrics.csv --events data/events.json
    mar attribute --data data/ad_metrics.csv --dim keyword_pack --metric cpa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import anomalies, attribution, benchmark, plotting, synth


def _cmd_synth(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = synth.generate(days=args.days, seed=args.seed)
    path = out / "ad_metrics.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df):,} rows, {df['date'].nunique()} days)")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=["date"])
    events = anomalies.events_from_dicts(json.loads(Path(args.events).read_text("utf-8")))

    result = benchmark.run(df, events, tolerance=args.tolerance)
    result.table.to_csv(out / "benchmark.csv", index=False)
    md = benchmark.to_markdown(result.table)
    (out / "benchmark.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def _cmd_attribute(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.data, parse_dates=["date"])
    last = df["date"].max()
    cut = last - pd.Timedelta(days=args.window - 1)
    prev_cut = cut - pd.Timedelta(days=args.window)

    current = df[df["date"] >= cut]
    baseline = df[(df["date"] >= prev_cut) & (df["date"] < cut)]

    table = attribution.shapley_change(baseline, current, by=args.dim, metric=args.metric)
    print(attribution.explain(baseline, current, by=args.dim, metric=args.metric))
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


def _cmd_demo(args: argparse.Namespace) -> None:
    root = Path(args.out)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)

    print("[1/5] generating synthetic delivery data ...")
    clean = synth.generate(days=args.days, seed=args.seed)
    contaminated, events = anomalies.inject(clean)

    (root / "data" / "ad_metrics.csv").parent.mkdir(parents=True, exist_ok=True)
    contaminated.to_csv(root / "data" / "ad_metrics.csv", index=False)
    (root / "data" / "events.json").write_text(
        json.dumps(anomalies.events_to_dicts(events), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"      {len(contaminated):,} rows, {len(events)} injected events")

    print("[2/5] running detectors ...")
    result = benchmark.run(contaminated, events, tolerance=args.tolerance)
    result.table.to_csv(root / "results" / "benchmark.csv", index=False)
    markdown = benchmark.to_markdown(result.table)
    (root / "results" / "benchmark.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)

    print("[3/5] plotting one case ...")
    series = anomalies.monitored_series(contaminated)["pack:high_intent_generic"]
    case_events = [
        e
        for e in events
        if e.series == "pack:high_intent_generic" or e.scope == "all"
    ]
    plotting.plot_example(
        series,
        {name: flags["pack:high_intent_generic"] for name, flags in result.flags.items()},
        case_events,
        root / "results" / "figures" / "detection_example.png",
        title="Budget rule change on one keyword pack: alarms vs. ground truth",
    )

    print("[4/5] attributing the CPA change ...")
    # Attribute the level-shift event, not an arbitrary calendar window: the
    # question is "which slice moved the metric while the budget rule changed".
    event = next(e for e in events if e.kind == "level_shift" and e.series.startswith("pack:"))
    start, end = pd.Timestamp(event.start), pd.Timestamp(event.end)
    current = contaminated[(contaminated["date"] >= start) & (contaminated["date"] <= end)]
    baseline = contaminated[
        (contaminated["date"] >= start - pd.Timedelta(days=28)) & (contaminated["date"] < start)
    ]
    table = attribution.shapley_change(baseline, current, by="keyword_pack", metric="cpa")
    table.to_csv(root / "results" / "attribution.csv", index=False)
    plotting.plot_attribution(table, root / "results" / "figures" / "attribution.png")
    print("      " + attribution.explain(baseline, current, by="keyword_pack", metric="cpa"))

    print("[5/5] done.")
    print(f"      benchmark -> {root / 'results' / 'benchmark.md'}")
    print(f"      figures   -> {root / 'results' / 'figures'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mar", description="Metric anomaly radar")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="end-to-end run with synthetic data")
    p_demo.add_argument("--out", default=".", help="output root directory")
    p_demo.add_argument("--days", type=int, default=180)
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.add_argument("--tolerance", type=int, default=3, help="detection window in days")
    p_demo.set_defaults(func=_cmd_demo)

    p_synth = sub.add_parser("synth", help="write a clean synthetic dataset")
    p_synth.add_argument("--out", default="data")
    p_synth.add_argument("--days", type=int, default=180)
    p_synth.add_argument("--seed", type=int, default=7)
    p_synth.set_defaults(func=_cmd_synth)

    p_bench = sub.add_parser("benchmark", help="score all detectors against the answer key")
    p_bench.add_argument("--data", default="data/ad_metrics.csv")
    p_bench.add_argument("--events", default="data/events.json")
    p_bench.add_argument("--out", default="results")
    p_bench.add_argument("--tolerance", type=int, default=3)
    p_bench.set_defaults(func=_cmd_benchmark)

    p_attr = sub.add_parser("attribute", help="Shapley attribution of a metric change")
    p_attr.add_argument("--data", default="data/ad_metrics.csv")
    p_attr.add_argument("--dim", default="keyword_pack")
    p_attr.add_argument("--metric", default="cpa", choices=["cpa", "spend", "clicks", "conversions"])
    p_attr.add_argument("--window", type=int, default=28)
    p_attr.set_defaults(func=_cmd_attribute)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
