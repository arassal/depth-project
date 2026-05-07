from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

CURRENT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry).resolve() != CURRENT_DIR]

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an indoor-only unlabeled image pool for teacher pseudo-labeling.")
    parser.add_argument("--root", required=True, help="Output directory for exported RGB images")
    parser.add_argument("--limit", type=int, default=2000, help="Number of indoor images to export")
    parser.add_argument(
        "--repo-id",
        default="prithivMLmods/IndoorOutdoorNet-20K",
        help="Hugging Face dataset repo containing an image parquet shard",
    )
    parser.add_argument("--filename", default="datasets/0000.parquet", help="Parquet filename inside the dataset repo")
    args = parser.parse_args()

    output_root = Path(args.root)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=args.filename)
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    label_names = dataset.features["label"].names
    if "Indoor" not in label_names:
        raise ValueError(f"Could not find 'Indoor' label in {label_names}")
    indoor_label = label_names.index("Indoor")

    kept = 0
    for index, row in enumerate(dataset):
        if row["label"] != indoor_label:
            continue
        row["image"].convert("RGB").save(output_root / f"{kept:06d}.jpg", quality=95)
        kept += 1
        if kept >= args.limit:
            break

    manifest_path = output_root / "export_summary.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"repo_id={args.repo_id}",
                f"filename={args.filename}",
                f"label=Indoor",
                f"num_images={kept}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print({"num_images": kept, "output_root": str(output_root)})


if __name__ == "__main__":
    main()
