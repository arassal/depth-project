from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def configure_epoch_axis(ax, epochs: pd.Series) -> None:
    values = epochs.tolist()
    ax.set_xticks(values)
    if len(values) == 1:
        epoch = float(values[0])
        ax.set_xlim(epoch - 0.5, epoch + 0.5)


def plot_series(ax, epochs: pd.Series, values: pd.Series, label: str | None = None) -> None:
    kwargs = {"marker": "o", "linewidth": 2, "markersize": 7}
    if label is not None:
        kwargs["label"] = label
    ax.plot(epochs, values, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-log", required=True)
    parser.add_argument("--per-image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame(read_jsonl(Path(args.metrics_log)))
    per_image = pd.read_csv(args.per_image)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_series(ax, history["epoch"], history["train_loss"], label="train")
    plot_series(ax, history["epoch"], history["val_loss"], label="val")
    configure_epoch_axis(ax, history["epoch"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_epoch.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_series(ax, history["epoch"], history["val_abs_rel"])
    configure_epoch_axis(ax, history["epoch"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AbsRel")
    ax.set_title("Validation AbsRel vs Epoch")
    fig.tight_layout()
    fig.savefig(output_dir / "val_absrel_vs_epoch.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_series(ax, history["epoch"], history["val_delta1"])
    configure_epoch_axis(ax, history["epoch"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("delta1")
    ax.set_title("Validation delta1 vs Epoch")
    fig.tight_layout()
    fig.savefig(output_dir / "val_delta1_vs_epoch.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_image["abs_rel"], bins=30)
    ax.set_xlabel("Per-image AbsRel")
    ax.set_ylabel("Count")
    ax.set_title("Test Per-image AbsRel Histogram")
    fig.tight_layout()
    fig.savefig(output_dir / "test_absrel_histogram.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
