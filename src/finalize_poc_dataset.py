from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def move_partition(root: Path, split_name: str, files: list[Path], start_index: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    image_target = root / "images" / split_name
    depth_target = root / "depth" / split_name
    image_target.mkdir(parents=True, exist_ok=True)
    depth_target.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(files, start=start_index):
        depth_path = root / "depth" / "train" / f"{image_path.stem}.npy"
        new_image = image_target / f"{index:06d}.png"
        new_depth = depth_target / f"{index:06d}.npy"
        shutil.move(str(image_path), str(new_image))
        shutil.move(str(depth_path), str(new_depth))
        rows.append(
            (
                new_image.relative_to(root).as_posix(),
                new_depth.relative_to(root).as_posix(),
            )
        )
    return rows


def write_split(root: Path, split_name: str, rows: list[tuple[str, str]]) -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / f"{split_name}.txt").open("w", encoding="utf-8") as handle:
        for image_rel, depth_rel in rows:
            handle.write(f"{image_rel} {depth_rel}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/alexander/depth-project/data/nyu_v2_poc")
    parser.add_argument("--train-count", type=int, default=450)
    parser.add_argument("--val-count", type=int, default=60)
    parser.add_argument("--test-count", type=int, default=59)
    args = parser.parse_args()

    root = Path(args.root)
    train_dir = root / "images" / "train"
    files = sorted(train_dir.glob("*.png"))
    needed = args.train_count + args.val_count + args.test_count
    if len(files) < needed:
        raise ValueError(f"Need at least {needed} image files, found {len(files)}")

    train_files = files[: args.train_count]
    val_files = files[args.train_count : args.train_count + args.val_count]
    test_files = files[args.train_count + args.val_count : needed]

    train_rows = [(p.relative_to(root).as_posix(), (root / "depth" / "train" / f"{p.stem}.npy").relative_to(root).as_posix()) for p in train_files]
    val_rows = move_partition(root, "val", val_files, 0)
    test_rows = move_partition(root, "test", test_files, 0)

    write_split(root, "train", train_rows)
    write_split(root, "val", val_rows)
    write_split(root, "test", test_rows)

    print(
        {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "root": str(root),
        }
    )


if __name__ == "__main__":
    main()
