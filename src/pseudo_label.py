from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import get_device, load_config, write_json
from modeling import forward_depth, load_model


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def sample_id_from_path(root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root)
    safe = "__".join(relative.with_suffix("").parts)
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    _, model = load_model(config["model"])

    teacher_mode = config["pseudo"].get("teacher_mode", "checkpoint")
    checkpoint_path = args.checkpoint or config["pseudo"].get("teacher_checkpoint")
    if teacher_mode == "checkpoint":
        if not checkpoint_path:
            raise ValueError("teacher_mode=checkpoint requires a checkpoint path")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    unlabeled_root = Path(config["data"]["unlabeled_root"])
    source_glob = config["pseudo"]["source_glob"]
    output_root = Path(config["pseudo"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    num_images = 0
    with torch.no_grad():
        for image_path in tqdm(sorted(unlabeled_root.rglob(source_glob)), desc="pseudo", leave=False):
            pixel_values = load_image(image_path, config["model"]["image_size"]).unsqueeze(0).to(device)
            pred = forward_depth(model, pixel_values)[0].cpu().numpy().astype(np.float32)
            sample_id = sample_id_from_path(unlabeled_root, image_path)
            depth_path = output_root / f"{sample_id}.npy"
            np.save(depth_path, pred)
            records.append(f"{image_path} {depth_path}")
            num_images += 1

    manifest_path = Path(config["data"]["pseudo_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(records) + ("\n" if records else ""))

    summary_path = config["pseudo"].get("summary_path")
    if summary_path:
        write_json(
            summary_path,
            {
                "teacher_mode": teacher_mode,
                "teacher_checkpoint": checkpoint_path,
                "unlabeled_root": str(unlabeled_root),
                "num_images": num_images,
                "output_root": str(output_root),
                "manifest_path": str(manifest_path),
            },
        )
    print({"teacher_mode": teacher_mode, "num_images": num_images, "manifest_path": str(manifest_path)})


if __name__ == "__main__":
    main()
