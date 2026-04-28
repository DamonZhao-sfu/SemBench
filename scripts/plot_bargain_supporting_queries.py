#!/usr/bin/env python3
"""Plot supporting-query figures for BARGAIN-R results.

Figure 1: recall target (x) vs precision (y)
Figure 2: recall target (x) vs number of calls saved (y)

Expected input columns (CSV):
- recall_target (required)
- precision (required)
- calls_saved (required)
- query_id (optional; when present, plots per-query scatter + mean line)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BARGAIN-R supporting-query figures")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Output directory")
    parser.add_argument("--title-prefix", type=str, default="BARGAIN-R", help="Plot title prefix")
    return parser.parse_args()


def _validate(df: pd.DataFrame) -> None:
    required = {"recall_target", "precision", "calls_saved"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("recall_target", as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            calls_saved_mean=("calls_saved", "mean"),
        )
        .sort_values("recall_target")
    )


def _plot_metric(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    y_col: str,
    y_label: str,
    outpath: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 5))

    if "query_id" in df.columns:
        for qid, group in df.groupby("query_id"):
            group = group.sort_values("recall_target")
            plt.plot(
                group["recall_target"],
                group[y_col],
                marker="o",
                alpha=0.25,
                linewidth=1,
                label=f"query={qid}",
            )

    plt.plot(
        agg["recall_target"],
        agg[f"{y_col}_mean"],
        marker="o",
        linewidth=2.5,
        color="black",
        label="mean",
    )

    plt.xlabel("Recall Target")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        if len(labels) > 8:
            # Too many query lines; keep only mean in legend for readability.
            plt.legend([handles[-1]], [labels[-1]], loc="best")
        else:
            plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    _validate(df)
    agg = _aggregate(df)

    _plot_metric(
        df,
        agg,
        y_col="precision",
        y_label="Precision",
        outpath=args.output_dir / "figure1_recall_target_vs_precision.png",
        title=f"{args.title_prefix}: Recall Target vs Precision",
    )

    _plot_metric(
        df,
        agg,
        y_col="calls_saved",
        y_label="Number of Calls Saved",
        outpath=args.output_dir / "figure2_recall_target_vs_calls_saved.png",
        title=f"{args.title_prefix}: Recall Target vs Calls Saved",
    )

    agg.to_csv(args.output_dir / "aggregated_metrics_by_recall_target.csv", index=False)
    print(f"Saved plots and aggregate table to: {args.output_dir}")


if __name__ == "__main__":
    main()
