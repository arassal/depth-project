from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from common import append_jsonl, get_device, load_config, seed_everything
from datasets import DepthDataset
from metrics import abs_rel, affine_invariant_l1, align_scale_and_shift, delta1, grad_l1_loss
from modeling import forward_depth, load_model


def build_loader(config: dict, split_key: str, shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    dataset = DepthDataset(
        root=data_cfg["root"],
        split_file=data_cfg[split_key],
        image_size=model_cfg["image_size"],
        min_depth=data_cfg["min_depth"],
        max_depth=data_cfg["max_depth"],
        image_mean=tuple(model_cfg.get("image_mean", (0.485, 0.456, 0.406))),
        image_std=tuple(model_cfg.get("image_std", (0.229, 0.224, 0.225))),
        has_depth=True,
        is_pseudo=False,
    )
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def maybe_build_pseudo_dataset(config: dict) -> DepthDataset | None:
    data_cfg = config["data"]
    pseudo_manifest = data_cfg.get("pseudo_manifest")
    if not pseudo_manifest or not Path(pseudo_manifest).exists():
        return None
    return DepthDataset(
        root=data_cfg["pseudo_root"],
        split_file=pseudo_manifest,
        image_size=config["model"]["image_size"],
        min_depth=data_cfg["min_depth"],
        max_depth=data_cfg["max_depth"],
        image_mean=tuple(config["model"].get("image_mean", (0.485, 0.456, 0.406))),
        image_std=tuple(config["model"].get("image_std", (0.229, 0.224, 0.225))),
        has_depth=True,
        is_pseudo=True,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    abs_rel_values = []
    delta1_values = []
    loss_values = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        pred = forward_depth(model, pixel_values)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        loss_values.append(affine_invariant_l1(pred, depth, valid_mask).item())
        aligned = align_scale_and_shift(pred, depth, valid_mask)
        abs_rel_values.append(abs_rel(aligned, depth, valid_mask).item())
        delta1_values.append(delta1(aligned, depth, valid_mask).item())
    return {
        "loss": sum(loss_values) / max(len(loss_values), 1),
        "abs_rel": sum(abs_rel_values) / max(len(abs_rel_values), 1),
        "delta1": sum(delta1_values) / max(len(delta1_values), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny CPU-only loop with no checkpointing (for test gates).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override epoch loop with a flat step budget.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    # --smoke-test forces a tiny CPU run with no checkpointing
    if args.smoke_test:
        device = torch.device("cpu")
        config["model"]["image_size"] = min(int(config["model"].get("image_size", 64)), 64)
        if args.max_steps is None:
            args.max_steps = 5
    else:
        device = get_device()

    # Distillation hook (Distill-Any-Depth He et al. 2025)
    distill_cfg = config.get("distillation", {}) or {}
    distill_enabled: bool = bool(distill_cfg.get("enabled", False))
    distill_model_id: str = distill_cfg.get("model_id", "LiheYoung/depth-anything-large-hf")
    distill_alpha: float = float(distill_cfg.get("alpha", 0.5))

    # Multi-scale gradient loss (ZoeDepth Bhat et al. 2023)
    loss_cfg = config.get("loss", {}) or {}
    grad_weight: float = float(loss_cfg.get("grad_weight", 0.0))

    _, model = load_model(config["model"]["pretrained_name"])
    model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Distillation hook (Distill-Any-Depth He et al. 2025): build teacher once, freeze it
    teacher = None
    if distill_enabled:
        from distill import build_teacher, teacher_predict, distillation_loss
        teacher, _ = build_teacher(distill_model_id, device, dtype=torch.float32)

    train_loader = build_loader(config, "train_split", shuffle=True)
    val_loader = build_loader(config, "val_split", shuffle=False)

    pseudo_dataset = maybe_build_pseudo_dataset(config)
    if pseudo_dataset is not None:
        train_dataset = ConcatDataset([train_loader.dataset, pseudo_dataset])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=config["train"]["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    best_abs_rel = float("inf")
    history_path = config["train"]["history_path"]
    save_path = Path(config["train"]["save_path"])
    if not args.smoke_test:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    max_steps = args.max_steps
    epochs = 1 if max_steps is not None else config["train"]["epochs"]
    stop = False

    for epoch in range(1, epochs + 1):
        if stop:
            break
        model.train()
        running_loss = 0.0
        running_base = 0.0
        running_distill = 0.0
        running_grad = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = forward_depth(model, pixel_values)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                base = affine_invariant_l1(pred, depth, valid_mask)
                d_val = 0.0
                g_val = 0.0

                # Distillation hook (Distill-Any-Depth He et al. 2025)
                if distill_enabled and teacher is not None:
                    with torch.no_grad():
                        teacher_pred = teacher_predict(
                            teacher, pixel_values, target_size=pred.shape[-2:]
                        )
                    d = distillation_loss(pred, teacher_pred, valid_mask)
                    loss = (1.0 - distill_alpha) * base + distill_alpha * d
                    d_val = float(d.detach().item())
                else:
                    loss = base

                # Multi-scale gradient loss (ZoeDepth Bhat et al. 2023)
                if grad_weight > 0.0:
                    g = grad_l1_loss(pred, depth, valid_mask)
                    loss = loss + grad_weight * g
                    g_val = float(g.detach().item())

            pseudo_weight = config["train"].get("pseudo_weight")
            if pseudo_weight is not None and "is_pseudo" in batch:
                batch_weights = torch.where(
                    torch.as_tensor(batch["is_pseudo"], device=device),
                    torch.full((len(batch["is_pseudo"]),), pseudo_weight, device=device),
                    torch.ones(len(batch["is_pseudo"]), device=device),
                )
                loss = loss * batch_weights.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_base += float(base.detach().item())
            running_distill += d_val
            running_grad += g_val
            global_step += 1
            if step % config["train"]["log_every"] == 0:
                progress.set_postfix(
                    loss=running_loss / step,
                    base=running_base / step,
                    distill=running_distill / step,
                    grad=running_grad / step,
                )

            if max_steps is not None and global_step >= max_steps:
                stop = True
                break

        denom = max(step if max_steps is not None else len(train_loader), 1)
        train_loss = running_loss / denom
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_base_loss": running_base / denom,
            "train_distill_loss": running_distill / denom,
            "train_grad_loss": running_grad / denom,
        }

        if not args.smoke_test:
            metrics = evaluate(model, val_loader, device)
            record.update({
                "val_loss": metrics["loss"],
                "val_abs_rel": metrics["abs_rel"],
                "val_delta1": metrics["delta1"],
            })
            append_jsonl(history_path, record)

            if metrics["abs_rel"] < best_abs_rel:
                best_abs_rel = metrics["abs_rel"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "val_abs_rel": best_abs_rel,
                    },
                    save_path,
                )

        print(record)


if __name__ == "__main__":
    main()
