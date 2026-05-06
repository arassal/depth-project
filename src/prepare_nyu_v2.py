from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm


DATASET_ID = "sayakpaul/nyu_depth_v2"


def decode_h5_bytes(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(io.BytesIO(payload), "r") as handle:
        rgb = np.array(handle["rgb"])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(handle["depth"]).astype(np.float32)
    return rgb, depth


def save_sample(root: Path, split_name: str, index: int, rgb: np.ndarray, depth: np.ndarray) -> tuple[str, str]:
    image_rel = Path("images") / split_name / f"{index:06d}.png"
    depth_rel = Path("depth") / split_name / f"{index:06d}.npy"

    image_path = root / image_rel
    depth_path = root / depth_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb.astype(np.uint8)).save(image_path)
    np.save(depth_path, depth.astype(np.float32))
    return image_rel.as_posix(), depth_rel.as_posix()


def write_manifest(root: Path, split_name: str, rows: list[tuple[str, str]]) -> None:
    split_path = root / "splits" / f"{split_name}.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as handle:
        for image_rel, depth_rel in rows:
            handle.write(f"{image_rel} {depth_rel}\n")


def download_shards(repo_id: str, train_shards: int | None, test_shards: int | None) -> dict[str, list[Path]]:
    files = list_repo_files(repo_id, repo_type="dataset")
    train_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".tar")])
    val_files = sorted([f for f in files if f.startswith("data/val-") and f.endswith(".tar")])

    if train_shards is not None:
        train_files = train_files[:train_shards]
    if test_shards is not None:
        val_files = val_files[:test_shards]

    shard_paths: dict[str, list[Path]] = {"train": [], "test": []}
    for split_name, split_files in [("train", train_files), ("test", val_files)]:
        for remote_path in tqdm(split_files, desc=f"download {split_name}", leave=False):
            local_path = hf_hub_download(repo_id=repo_id, filename=remote_path, repo_type="dataset")
            shard_paths[split_name].append(Path(local_path))
    return shard_paths


def iter_samples(shard_paths: list[Path]):
    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r") as archive:
            members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".h5")]
            for member in members:
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue
                yield file_obj.read()


def export_dataset(
    root: Path,
    val_count: int,
    train_shards: int | None,
    test_shards: int | None,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> None:
    shard_paths = download_shards(DATASET_ID, train_shards=train_shards, test_shards=test_shards)
    train_rows: list[tuple[str, str]] = []
    val_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []

    train_payloads = iter_samples(shard_paths["train"])
    buffered_train: list[tuple[np.ndarray, np.ndarray]] = []
    exported_train = 0
    train_limit = max_train_samples if max_train_samples is not None else float("inf")
    for payload in tqdm(train_payloads, desc="export train+val", leave=False):
        if exported_train >= train_limit:
            break
        rgb, depth = decode_h5_bytes(payload)
        buffered_train.append((rgb, depth))
        exported_train += 1

    if val_count <= 0 or val_count >= len(buffered_train):
        raise ValueError(f"val_count must be between 1 and {len(buffered_train) - 1}")

    split_boundary = len(buffered_train) - val_count
    for index, (rgb, depth) in enumerate(buffered_train):
        split_name = "train" if index < split_boundary else "val"
        rels = save_sample(root, split_name, index if split_name == "train" else index - split_boundary, rgb, depth)
        if split_name == "train":
            train_rows.append(rels)
        else:
            val_rows.append(rels)

    test_limit = max_test_samples if max_test_samples is not None else float("inf")
    for index, payload in enumerate(tqdm(iter_samples(shard_paths["test"]), desc="export test", leave=False)):
        if index >= test_limit:
            break
        rgb, depth = decode_h5_bytes(payload)
        test_rows.append(save_sample(root, "test", index, rgb, depth))

    write_manifest(root, "train", train_rows)
    write_manifest(root, "val", val_rows)
    write_manifest(root, "test", test_rows)

    print(
        {
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "root": str(root),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/alexander/depth-project/data/nyu_v2")
    parser.add_argument("--val-count", type=int, default=512)
    parser.add_argument("--train-shards", type=int, default=None)
    parser.add_argument("--test-shards", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    export_dataset(
        Path(args.root),
        val_count=args.val_count,
        train_shards=args.train_shards,
        test_shards=args.test_shards,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()
