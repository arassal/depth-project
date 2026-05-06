from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from common import append_jsonl, get_device, load_config, seed_everything
from datasets import DepthDataset
from metrics import abs_rel, affine_invariant_l1, align_scale_and_shift, delta1
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
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = get_device()

    _, model = load_model(config["model"]["pretrained_name"])
    model.to(device)

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
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = forward_depth(model, pixel_values)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            loss = affine_invariant_l1(pred, depth, valid_mask)
            pseudo_weight = config["train"].get("pseudo_weight")
            if pseudo_weight is not None and "is_pseudo" in batch:
                batch_weights = torch.where(
                    torch.as_tensor(batch["is_pseudo"], device=device),
                    torch.full((len(batch["is_pseudo"]),), pseudo_weight, device=device),
                    torch.ones(len(batch["is_pseudo"]), device=device),
                )
                loss = loss * batch_weights.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_norm"])
            optimizer.step()

            running_loss += loss.item()
            if step % config["train"]["log_every"] == 0:
                progress.set_postfix(loss=running_loss / step)

        train_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_abs_rel": metrics["abs_rel"],
            "val_delta1": metrics["delta1"],
        }
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
