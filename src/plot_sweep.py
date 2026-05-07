from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return pd.DataFrame(rows)


def line_plot(df: pd.DataFrame, x: str, y: str, output: Path, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[x], df[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xscale("log") if df[x].min() > 0 and df[x].max() / df[x].min() > 50 else None
    ax.set_title(title or f"{y} vs {x}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def heatmap(df: pd.DataFrame, x: str, y: str, value: str, output: Path) -> None:
    pivot = df.pivot(index=y, columns=x, values=value)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{value} ({y} vs {x})")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to sweep results.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--x", default=None, help="Column for x-axis (line) or heatmap x")
    parser.add_argument("--y", default=None, help="Heatmap y-axis column (omit for line plot)")
    parser.add_argument("--metric", default="abs_rel", help="Metric column to plot")
    args = parser.parse_args()

    df = load_results(Path(args.results))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.x is None:
        raise SystemExit("must pass --x")

    if args.y is None:
        line_plot(df.sort_values(args.x), args.x, args.metric, output_dir / f"{args.metric}_vs_{args.x}.png")
    else:
        heatmap(df, args.x, args.y, args.metric, output_dir / f"{args.metric}_{args.y}_vs_{args.x}.png")


if __name__ == "__main__":
    main()
