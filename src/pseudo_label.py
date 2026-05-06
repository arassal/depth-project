from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import get_device, load_config
from modeling import forward_depth, load_model


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    _, model = load_model(config["model"]["pretrained_name"])

    checkpoint_path = args.checkpoint or config["pseudo"]["teacher_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    unlabeled_root = Path(config["data"]["unlabeled_root"])
    source_glob = config["pseudo"]["source_glob"]
    output_root = Path(config["pseudo"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    with torch.no_grad():
        for image_path in tqdm(sorted(unlabeled_root.rglob(source_glob)), desc="pseudo", leave=False):
            pixel_values = load_image(image_path, config["model"]["image_size"]).unsqueeze(0).to(device)
            pred = forward_depth(model, pixel_values)[0].cpu().numpy().astype(np.float32)
            depth_path = output_root / f"{image_path.stem}.npy"
            np.save(depth_path, pred)
            records.append(f"{image_path} {depth_path}")

    manifest_path = Path(config["data"]["pseudo_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(records))


if __name__ == "__main__":
    main()
