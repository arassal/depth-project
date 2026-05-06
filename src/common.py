from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
