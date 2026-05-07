from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class ResidualRefinementHead(nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, rgb_and_depth: torch.Tensor) -> torch.Tensor:
        return self.net(rgb_and_depth)


class RefinedDepthModel(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        refinement_hidden_channels: int = 32,
        residual_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.base_model = AutoModelForDepthEstimation.from_pretrained(pretrained_name)
        self.refinement_head = ResidualRefinementHead(hidden_channels=refinement_hidden_channels)
        self.residual_scale = residual_scale

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(pixel_values=pixel_values)
        coarse_depth = outputs.predicted_depth
        if coarse_depth.ndim == 4 and coarse_depth.shape[1] == 1:
            coarse_depth = coarse_depth[:, 0]

        coarse_depth = F.interpolate(
            coarse_depth.unsqueeze(1),
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refinement_input = torch.cat([pixel_values, coarse_depth], dim=1)
        residual = self.refinement_head(refinement_input)
        refined = coarse_depth + self.residual_scale * residual
        return refined[:, 0]


def load_model(model_cfg: dict[str, Any] | str) -> tuple[AutoImageProcessor, nn.Module]:
    if isinstance(model_cfg, str):
        pretrained_name = model_cfg
        architecture = "baseline"
        refinement_hidden_channels = 32
        residual_scale = 0.15
    else:
        pretrained_name = model_cfg["pretrained_name"]
        architecture = model_cfg.get("architecture", "baseline")
        refinement_hidden_channels = int(model_cfg.get("refinement_hidden_channels", 32))
        residual_scale = float(model_cfg.get("residual_scale", 0.15))

    processor = AutoImageProcessor.from_pretrained(pretrained_name)
    if architecture == "baseline":
        model: nn.Module = AutoModelForDepthEstimation.from_pretrained(pretrained_name)
    elif architecture == "rgb_residual_refinement":
        model = RefinedDepthModel(
            pretrained_name=pretrained_name,
            refinement_hidden_channels=refinement_hidden_channels,
            residual_scale=residual_scale,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return processor, model


def forward_depth(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = model(pixel_values=pixel_values) if not isinstance(model, RefinedDepthModel) else model(pixel_values)
    if isinstance(outputs, torch.Tensor):
        pred = outputs
    else:
        pred = outputs.predicted_depth
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred
