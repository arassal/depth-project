from __future__ import annotations

import argparse
from pathlib import Path


def write_split(root: Path, split_name: str, image_dir: str, depth_dir: str, output_path: Path) -> None:
    image_root = root / image_dir / split_name
    depth_root = root / depth_dir / split_name
    image_files = sorted(image_root.glob("*"))
    rows = []
    for image_path in image_files:
        if not image_path.is_file():
            continue
        stem = image_path.stem
        depth_candidates = [
            depth_root / f"{stem}.png",
            depth_root / f"{stem}.npy",
            depth_root / f"{stem}.tiff",
            depth_root / f"{stem}.tif",
        ]
        depth_path = next((candidate for candidate in depth_candidates if candidate.exists()), None)
        if depth_path is None:
            continue
        rows.append(f"{image_path.relative_to(root)} {depth_path.relative_to(root)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"{split_name}: wrote {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root, for example data/nyu_v2")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--depth-dir", default="depth")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(args.root)
    for split_name in args.splits:
        write_split(
            root=root,
            split_name=split_name,
            image_dir=args.image_dir,
            depth_dir=args.depth_dir,
            output_path=root / "splits" / f"{split_name}.txt",
        )


if __name__ == "__main__":
    main()
