from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_NAME = "LiheYoung/depth-anything-small-hf"


def colorize_depth(depth: np.ndarray) -> Image.Image:
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)
    colored = cm.get_cmap("plasma")(depth)[..., :3]
    return Image.fromarray((colored * 255).astype(np.uint8))


def iter_images(input_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(device).eval()

    images = list(iter_images(input_dir))
    if not images:
        raise ValueError(f"No images found under {input_dir}")

    with torch.no_grad():
        for image_path in tqdm(images, desc="infer", leave=False):
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            pred = model(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            pred = pred.detach().cpu().numpy()
            color = colorize_depth(pred)

            rel = image_path.relative_to(input_dir)
            out_path = output_dir / rel.with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            color.save(out_path)

    print({"num_images": len(images), "input_dir": str(input_dir), "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
