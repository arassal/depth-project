from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)
    return (cm.get_cmap("plasma")(depth)[..., :3] * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-count", type=int, required=True)
    parser.add_argument("--pseudo-summary", required=True)
    parser.add_argument("--pseudo-manifest", required=True)
    parser.add_argument("--output-plot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    pseudo_summary = json.loads(Path(args.pseudo_summary).read_text())
    pseudo_count = int(pseudo_summary["num_images"])
    total_count = args.labeled_count + pseudo_count

    plot_path = Path(args.output_plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Labeled seed", "Pseudo-labeled expansion", "Total student train"]
    values = [args.labeled_count, pseudo_count, total_count]
    bars = ax.bar(labels, values, color=["#b24a2a", "#d88558", "#2d6a4f"])
    ax.set_ylabel("Images")
    ax.set_title("Teacher-Student Data Growth")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    rows = [line.strip() for line in Path(args.pseudo_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows[: args.num_samples]):
        image_path_str, depth_path_str = row.split()
        image = Image.open(image_path_str).convert("RGB")
        depth = np.load(depth_path_str)
        color = Image.fromarray(colorize_depth(depth))

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image)
        axes[0].set_title("Unlabeled RGB")
        axes[1].imshow(color)
        axes[1].set_title("Teacher pseudo-depth")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{index:02d}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
