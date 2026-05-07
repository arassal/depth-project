"""Build a KITTI eval manifest in the project's split format.

Adapted from references/Depth-Anything-V2/metric_depth/dataset/kitti.py and
references/Depth-Anything-V2/metric_depth/dataset/splits/kitti/val.txt.

KITTI must be downloaded manually (~175 GB raw + ~14 GB depth annotated):
  Raw sequences:        http://www.cvlibs.net/datasets/kitti/raw_data.php
  Depth annotated:      https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction

Expected on-disk layout under --root:
    <root>/raw_data/<date>/<drive>/image_02/data/<frame>.png
    <root>/data_depth_annotated/val/<drive>/proj_depth/groundtruth/image_02/<frame>.png

The reference split file embeds the original mount prefix
``/mnt/bn/liheyang/Kitti``; we strip that and join under --root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Prefix used in the reference split file we lift sample IDs from.
REFERENCE_PREFIX = "/mnt/bn/liheyang/Kitti/"

DEFAULT_SPLIT_FILE = (
    Path(__file__).resolve().parent.parent
    / "references"
    / "Depth-Anything-V2"
    / "metric_depth"
    / "dataset"
    / "splits"
    / "kitti"
    / "val.txt"
)

DOWNLOAD_INSTRUCTIONS = """\
KITTI evaluation data was not found at {root}.

KITTI is not downloadable via huggingface_hub; you must register and fetch it
manually:

  1. Raw sequences (image_02 RGB):
       http://www.cvlibs.net/datasets/kitti/raw_data.php
     Download the synced+rectified drives referenced in val.txt and unpack
     them so they live at:
       {root}/raw_data/<date>/<drive>/image_02/data/*.png

  2. Depth annotated (Eigen test ground truth):
       https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
     Unpack so the val ground-truth lives at:
       {root}/data_depth_annotated/val/<drive>/proj_depth/groundtruth/image_02/*.png

Re-run this script once both trees exist.
"""


def _strip_prefix(path: str) -> str:
    if path.startswith(REFERENCE_PREFIX):
        return path[len(REFERENCE_PREFIX):]
    # Fallback: also tolerate a leading slash so callers can pass already-relative paths.
    return path.lstrip("/")


def build_manifest(
    root: Path,
    split_file: Path,
    output_path: Path,
    limit: int | None = None,
    require_files: bool = True,
) -> int:
    if not split_file.exists():
        raise FileNotFoundError(f"reference split file missing: {split_file}")

    rows: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []

    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            image_rel = _strip_prefix(parts[0])
            depth_rel = _strip_prefix(parts[1])

            if require_files:
                if not (root / image_rel).exists() or not (root / depth_rel).exists():
                    missing.append((image_rel, depth_rel))
                    continue

            rows.append((image_rel, depth_rel))
            if limit is not None and len(rows) >= limit:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for image_rel, depth_rel in rows:
            handle.write(f"{image_rel} {depth_rel}\n")

    if missing:
        print(f"warning: skipped {len(missing)} samples whose files are missing under {root}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KITTI eval manifest from the eigen val split.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root that contains raw_data/ and data_depth_annotated/ KITTI trees.",
    )
    parser.add_argument(
        "--split-file",
        default=str(DEFAULT_SPLIT_FILE),
        help="Reference split file to lift sample IDs from.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output manifest path (default: <root>/manifests/test.txt).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of manifest entries (useful for fast iteration).",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Do not verify each file exists on disk; emit every line from the reference split.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(DOWNLOAD_INSTRUCTIONS.format(root=root))
        raise SystemExit(2)

    output_path = Path(args.output) if args.output else root / "manifests" / "test.txt"
    split_file = Path(args.split_file)

    written = build_manifest(
        root=root,
        split_file=split_file,
        output_path=output_path,
        limit=args.limit,
        require_files=not args.no_check,
    )
    print({"manifest": str(output_path), "samples": written, "root": str(root)})


if __name__ == "__main__":
    main()
