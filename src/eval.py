from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import get_device, load_config, write_json
from datasets import DepthDataset
from metrics import abs_rel, align_scale_and_shift, delta1
from modeling import forward_depth, load_model


def save_preview(output_dir: Path, sample_id: str, rgb: torch.Tensor, depth: torch.Tensor, pred: torch.Tensor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    error = (pred - depth).abs()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(rgb.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("RGB")
    axes[1].imshow(depth.cpu().numpy(), cmap="plasma")
    axes[1].set_title("GT")
    axes[2].imshow(pred.cpu().numpy(), cmap="plasma")
    axes[2].set_title("Pred")
    axes[3].imshow(error.cpu().numpy(), cmap="inferno")
    axes[3].set_title("Error")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{sample_id}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--per-image-path", default=None)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--num-preview-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    _, model = load_model(config["model"])

    checkpoint_path = args.checkpoint if args.checkpoint is not None else config["eval"].get("checkpoint")
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    use_amp = device.type == "cuda"

    dataset = DepthDataset(
        root=config["data"]["root"],
        split_file=config["data"]["test_split"],
        image_size=config["model"]["image_size"],
        min_depth=config["data"]["min_depth"],
        max_depth=config["data"]["max_depth"],
        image_mean=tuple(config["model"].get("image_mean", (0.485, 0.456, 0.406))),
        image_std=tuple(config["model"].get("image_std", (0.229, 0.224, 0.225))),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rows = []
    preview_dir = Path(args.preview_dir or config["eval"]["preview_dir"])
    preview_count = 0
    preview_limit = args.num_preview_samples if args.num_preview_samples is not None else config["eval"]["num_preview_samples"]

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            sample_id = batch["sample_id"][0]

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = forward_depth(model, pixel_values)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            aligned = align_scale_and_shift(pred, depth, valid_mask)

            sample_abs_rel = abs_rel(aligned, depth, valid_mask).item()
            sample_delta1 = delta1(aligned, depth, valid_mask).item()
            rows.append({"sample_id": sample_id, "abs_rel": sample_abs_rel, "delta1": sample_delta1})

            if preview_limit < 0 or preview_count < preview_limit:
                save_preview(
                    preview_dir,
                    sample_id,
                    batch["rgb_vis"][0],
                    depth[0].cpu(),
                    aligned[0].cpu(),
                )
                preview_count += 1

    df = pd.DataFrame(rows)
    per_image_path = Path(args.per_image_path or config["eval"]["per_image_path"])
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_image_path, index=False)

    summary = {
        "abs_rel": float(df["abs_rel"].mean()),
        "delta1": float(df["delta1"].mean()),
        "num_samples": int(len(df)),
    }
    write_json(args.summary_path or config["eval"]["summary_path"], summary)
    print(summary)


if __name__ == "__main__":
    main()
