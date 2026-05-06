from __future__ import annotations

import torch


def robust_normalize(depth: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    values = depth[valid_mask]
    if values.numel() == 0:
        return torch.zeros_like(depth)
    median = values.median()
    spread = (values - median).abs().mean().clamp_min(eps)
    return (depth - median) / spread


def affine_invariant_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for sample_pred, sample_target, sample_mask in zip(pred, target, valid_mask):
        if sample_mask.sum() == 0:
            continue
        pred_norm = robust_normalize(sample_pred, sample_mask)
        target_norm = robust_normalize(sample_target, sample_mask)
        losses.append((pred_norm[sample_mask] - target_norm[sample_mask]).abs().mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def align_scale_and_shift(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    aligned = torch.zeros_like(pred)
    for index, (sample_pred, sample_target, sample_mask) in enumerate(zip(pred, target, valid_mask)):
        pred_values = sample_pred[sample_mask]
        target_values = sample_target[sample_mask]
        if pred_values.numel() < 2:
            aligned[index] = sample_pred
            continue
        a_00 = (pred_values * pred_values).sum()
        a_01 = pred_values.sum()
        a_11 = torch.tensor(float(pred_values.numel()), device=pred.device, dtype=pred.dtype)
        b_0 = (pred_values * target_values).sum()
        b_1 = target_values.sum()
        det = a_00 * a_11 - a_01 * a_01
        if det.abs() < 1e-6:
            scale = 1.0
            shift = 0.0
        else:
            scale = (a_11 * b_0 - a_01 * b_1) / det
            shift = (-a_01 * b_0 + a_00 * b_1) / det
        aligned[index] = scale * sample_pred + shift
    return aligned


def abs_rel(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    values = (pred[valid_mask] - target[valid_mask]).abs() / target[valid_mask].clamp_min(eps)
    return values.mean() if values.numel() else pred.new_tensor(float("nan"))


def delta1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_values = pred[valid_mask].clamp_min(eps)
    target_values = target[valid_mask].clamp_min(eps)
    if pred_values.numel() == 0:
        return pred.new_tensor(float("nan"))
    ratios = torch.maximum(pred_values / target_values, target_values / pred_values)
    return (ratios < 1.25).float().mean()
