"""Output-level distillation for monocular depth estimation.

Teacher-student setup adapted from Distill-Any-Depth (He et al. 2025,
arXiv 2502.19204). We implement only the simpler output-level
affine-invariant distillation; the original paper additionally uses
cross-context (global+local) crops which we omit for simplicity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from metrics import affine_invariant_l1


def build_teacher(
    model_id: str = "LiheYoung/depth-anything-large-hf",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, AutoImageProcessor]:
    """Load a frozen teacher depth model. Returns (model_in_eval_mode, image_processor)."""
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    processor = AutoImageProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def teacher_predict(
    teacher: torch.nn.Module,
    pixel_values: torch.Tensor,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Run teacher in no-grad, optionally upsample to match student output size."""
    outputs = teacher(pixel_values=pixel_values)
    pred = outputs.predicted_depth
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target_size is not None:
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return pred


def distillation_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Affine-invariant L1 between student and teacher predictions.

    Both predictions are robustly normalized (median + MAD) per-sample, then L1.
    If valid_mask is None, all pixels are used.
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(student_pred, dtype=torch.bool)
    return affine_invariant_l1(student_pred, teacher_pred, valid_mask)
