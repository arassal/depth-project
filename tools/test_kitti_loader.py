"""Smoke-test the KITTI depth scaling path of DepthDataset.

Synthesizes a tiny fake KITTI sample (one 16-bit PNG depth + one RGB image),
builds a manifest, and verifies DepthDataset returns a sample whose depth
values land in the 0-80 m range expected for KITTI.

Run:
    python tools/test_kitti_loader.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Make src/ importable when run from the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from datasets import DepthDataset  # noqa: E402


def _write_fake_sample(root: Path) -> tuple[str, str]:
    image_rel = Path("raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png")
    depth_rel = Path(
        "data_depth_annotated/val/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000000.png"
    )

    image_path = root / image_rel
    depth_path = root / depth_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)

    # Fake RGB: 64x64 random uint8.
    rgb = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(image_path)

    # Fake KITTI depth: 16-bit PNG, raw values in [0, 80*256] so meters land in [0, 80].
    raw = np.random.default_rng(1).integers(256, 80 * 256, size=(64, 64), dtype=np.uint16)
    Image.fromarray(raw, mode="I;16").save(depth_path)

    return image_rel.as_posix(), depth_rel.as_posix()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_rel, depth_rel = _write_fake_sample(root)

        manifest = root / "manifests" / "test.txt"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(f"{image_rel} {depth_rel}\n", encoding="utf-8")

        dataset = DepthDataset(
            root=root,
            split_file="manifests/test.txt",
            image_size=64,
            min_depth=0.001,
            max_depth=80.0,
            depth_scale=256.0,
        )

        assert len(dataset) == 1, f"expected 1 record, got {len(dataset)}"
        sample = dataset[0]

        for key in ("pixel_values", "depth", "valid_mask", "rgb_vis", "sample_id"):
            assert key in sample, f"missing key {key}"

        pixel_values = sample["pixel_values"]
        depth = sample["depth"]
        valid_mask = sample["valid_mask"]

        assert pixel_values.shape == (3, 64, 64), f"pixel shape {pixel_values.shape}"
        assert depth.shape == (64, 64), f"depth shape {depth.shape}"
        assert valid_mask.shape == (64, 64), f"mask shape {valid_mask.shape}"

        valid_depth = depth[valid_mask]
        assert valid_depth.numel() > 0, "no valid depth pixels"
        d_min = float(valid_depth.min())
        d_max = float(valid_depth.max())
        assert 0.0 < d_min, f"min depth not positive: {d_min}"
        assert d_max < 80.0, f"max depth out of range: {d_max}"

        # Sanity check: NYU path (no depth_scale) on the same 16-bit PNG should NOT
        # produce KITTI-style metres. This guards against breaking NYU-style loading.
        nyu_dataset = DepthDataset(
            root=root,
            split_file="manifests/test.txt",
            image_size=64,
            min_depth=0.001,
            max_depth=10.0,
            depth_scale=None,
        )
        nyu_sample = nyu_dataset[0]
        # /1000.0 heuristic on raw values up to ~80*256=20480 yields up to ~20.5
        assert float(nyu_sample["depth"].max()) > 1.0, "NYU heuristic appears not to fire"

    print(
        "PASS",
        {
            "depth_min_m": round(d_min, 4),
            "depth_max_m": round(d_max, 4),
            "valid_pixels": int(valid_mask.sum()),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
