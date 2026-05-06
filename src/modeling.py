from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def load_model(pretrained_name: str) -> tuple[AutoImageProcessor, AutoModelForDepthEstimation]:
    processor = AutoImageProcessor.from_pretrained(pretrained_name)
    model = AutoModelForDepthEstimation.from_pretrained(pretrained_name)
    return processor, model


def forward_depth(model: AutoModelForDepthEstimation, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = model(pixel_values=pixel_values)
    pred = outputs.predicted_depth
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    return pred
