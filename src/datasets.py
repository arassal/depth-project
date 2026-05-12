from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    image_path: Path
    depth_path: Path | None
    sample_id: str
    is_pseudo: bool = False


def _load_depth(path: Path, depth_scale: float | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = np.array(Image.open(path), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth_scale is not None:
            # Explicit scale path (e.g. KITTI uses /256.0 for meters).
            depth = depth / float(depth_scale)
        elif path.suffix.lower() in {".png", ".tif", ".tiff"} and depth.max() > 255:
            # Legacy NYU heuristic: 16-bit PNG in millimetres.
            depth = depth / 1000.0
    return depth.astype(np.float32)


def _resize_tensor(tensor: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    tensor = tensor.unsqueeze(0)
    if mode == "nearest":
        tensor = F.interpolate(tensor, size=(size, size), mode=mode)
    else:
        tensor = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=False)
    return tensor.squeeze(0)


class DepthDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_file: str | Path,
        image_size: int,
        min_depth: float,
        max_depth: float,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        has_depth: bool = True,
        is_pseudo: bool = False,
        depth_scale: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.has_depth = has_depth
        self.depth_scale = depth_scale
        self.records = self._load_records(split_file, is_pseudo=is_pseudo)

    def _load_records(self, split_file: str | Path, is_pseudo: bool) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        split_path = self.root / split_file if not Path(split_file).is_absolute() else Path(split_file)
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                image_path = Path(parts[0])
                if not image_path.is_absolute():
                    image_path = self.root / image_path
                depth_path = None
                if self.has_depth and len(parts) > 1:
                    depth_path = Path(parts[1])
                    if not depth_path.is_absolute():
                        depth_path = self.root / depth_path
                records.append(
                    SampleRecord(
                        image_path=image_path,
                        depth_path=depth_path,
                        sample_id=image_path.stem,
                        is_pseudo=is_pseudo,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        image_tensor = _resize_tensor(image_tensor, self.image_size, mode="bilinear")
        rgb_vis = image_tensor.clone()
        image_tensor = (image_tensor - self.image_mean) / self.image_std

        sample: dict[str, torch.Tensor | str | bool] = {
            "pixel_values": image_tensor,
            "rgb_vis": rgb_vis,
            "sample_id": record.sample_id,
            "is_pseudo": record.is_pseudo,
        }

        if record.depth_path is not None:
            depth_np = _load_depth(record.depth_path, depth_scale=self.depth_scale)
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
            depth_tensor = _resize_tensor(depth_tensor, self.image_size, mode="nearest")
            depth_tensor = depth_tensor.squeeze(0)
            valid_mask = torch.isfinite(depth_tensor)
            valid_mask &= depth_tensor > self.min_depth
            valid_mask &= depth_tensor < self.max_depth
            depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            sample["depth"] = depth_tensor
            sample["valid_mask"] = valid_mask

        return sample
