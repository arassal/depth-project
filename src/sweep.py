"""Hyperparameter sweep runner.

Reads a base training config and a sweep specification, generates a derived
config for every point in the cartesian product of the sweep grid, runs
``src/train.py`` followed by ``src/eval.py`` for each derived config, collects
per-run metrics from the eval summary JSON, and prints a sorted markdown table.

Usage:
    python -m src.sweep --base configs/distill.yaml \
        --grid configs/sweeps/lr_batch.yaml \
        --out results/sweeps/run1

Sweep spec schema::

    sweep:
      learning_rate: [1.0e-5, 5.0e-5, 1.0e-4]
      train_batch_size: [2, 4, 8]
    keys:
      learning_rate: "training.learning_rate"   # dot-notation into base config
      train_batch_size: "training.batch_size"
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``value`` at ``dotted_key`` (e.g. ``"train.learning_rate"``).

    Intermediate dict nodes are created as needed.
    """
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def build_overrides(
    sweep_keys: list[str],
    combo: tuple[Any, ...],
    key_map: dict[str, str],
) -> dict[str, Any]:
    """Return a dict of {dotted_path: value} for one grid point."""
    return {key_map[name]: combo[idx] for idx, name in enumerate(sweep_keys)}


def apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    derived = copy.deepcopy(base)
    for dotted, value in overrides.items():
        set_dotted(derived, dotted, value)
    return derived


def redirect_outputs(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Point train/eval output paths into the per-run directory.

    This keeps each sweep run isolated so that history / checkpoints / metrics
    don't clobber one another. Only sets keys whose parents already exist in
    the base config (so we don't invent unrelated structure).
    """
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts / "metrics.json"

    if isinstance(config.get("train"), dict):
        config["train"]["save_path"] = str(artifacts / "best.pt")
        config["train"]["history_path"] = str(artifacts / "history.jsonl")
    if isinstance(config.get("training"), dict):
        config["training"]["save_path"] = str(artifacts / "best.pt")
        config["training"]["history_path"] = str(artifacts / "history.jsonl")

    if isinstance(config.get("eval"), dict):
        config["eval"]["checkpoint"] = str(artifacts / "best.pt")
        config["eval"]["summary_path"] = str(summary_path)
        config["eval"]["per_image_path"] = str(artifacts / "per_image.csv")
        config["eval"]["preview_dir"] = str(artifacts / "previews")

    return config


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def run_subprocess(cmd: list[str], log_path: Path) -> int:
    """Run ``cmd`` writing combined stdout+stderr to ``log_path``.

    Returns the process return code. Never raises on non-zero exit — callers
    decide how to handle failure.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(  # noqa: S603 - intentional subprocess
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return completed.returncode


def find_metrics_json(config: dict[str, Any], run_dir: Path) -> Path | None:
    """Locate the metrics.json eval wrote.

    ``src/eval.py`` writes to ``config['eval']['summary_path']`` (see
    ``write_json(args.summary_path or config["eval"]["summary_path"], ...)``).
    We standardize that to ``<run_dir>/artifacts/metrics.json`` in
    ``redirect_outputs``, but fall back to the raw config value if a custom
    base config bypasses redirection.
    """
    candidate = run_dir / "artifacts" / "metrics.json"
    if candidate.exists():
        return candidate
    eval_cfg = config.get("eval") or {}
    summary_path = eval_cfg.get("summary_path")
    if summary_path and Path(summary_path).exists():
        return Path(summary_path)
    return None


def parse_metrics(metrics_path: Path | None) -> dict[str, Any]:
    if metrics_path is None:
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"_parse_error": str(exc)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_markdown_table(
    results: list[dict[str, Any]],
    sweep_keys: list[str],
) -> str:
    """Render a markdown results table sorted by AbsRel ascending."""
    if not results:
        return "_no results_"

    metric_keys: list[str] = []
    for record in results:
        for key in record.get("metrics", {}):
            if key.startswith("_"):
                continue
            if key not in metric_keys:
                metric_keys.append(key)

    # Put AbsRel and delta1 first when present, then any others.
    priority = [k for k in ("abs_rel", "delta1") if k in metric_keys]
    rest = [k for k in metric_keys if k not in priority]
    ordered_metrics = priority + rest

    def sort_key(record: dict[str, Any]) -> float:
        value = record.get("metrics", {}).get("abs_rel")
        return float(value) if isinstance(value, (int, float)) else float("inf")

    sorted_results = sorted(results, key=sort_key)

    header = ["run_idx", *sweep_keys, *ordered_metrics, "status"]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for record in sorted_results:
        row = [str(record["run_idx"])]
        for key in sweep_keys:
            row.append(str(record["config_overrides"].get(key, "")))
        for key in ordered_metrics:
            value = record.get("metrics", {}).get(key)
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append("" if value is None else str(value))
        row.append(record.get("status", ""))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep runner")
    parser.add_argument("--base", required=True, help="Path to base training config (yaml).")
    parser.add_argument("--grid", required=True, help="Path to sweep spec yaml (sweep + keys).")
    parser.add_argument("--out", required=True, help="Output directory for derived configs / logs / results.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands; do not execute.")
    parser.add_argument("--max-runs", type=int, default=None, help="Run only the first N combos.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for child train/eval commands.",
    )
    args = parser.parse_args()

    base_config = load_yaml(args.base)
    grid_spec = load_yaml(args.grid)

    sweep_grid = grid_spec.get("sweep") or {}
    key_map = grid_spec.get("keys") or {}
    if not sweep_grid:
        print("error: sweep spec is missing a non-empty 'sweep' block", file=sys.stderr)
        return 2
    missing = [name for name in sweep_grid if name not in key_map]
    if missing:
        print(f"error: 'keys' map is missing entries for: {missing}", file=sys.stderr)
        return 2

    sweep_keys = list(sweep_grid.keys())
    value_lists = [list(sweep_grid[name]) for name in sweep_keys]
    combos = list(itertools.product(*value_lists))
    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    out_dir = Path(args.out)
    configs_dir = out_dir / "configs"
    logs_dir = out_dir / "logs"

    results: list[dict[str, Any]] = []
    print(f"Planning {len(combos)} run(s) from grid: {sweep_keys}")

    for run_idx, combo in enumerate(combos):
        overrides_named = dict(zip(sweep_keys, combo))
        overrides_dotted = build_overrides(sweep_keys, combo, key_map)
        derived = apply_overrides(base_config, overrides_dotted)
        run_dir = out_dir / f"run_{run_idx}"
        derived = redirect_outputs(derived, run_dir)

        derived_path = configs_dir / f"run_{run_idx}.yaml"
        train_log = logs_dir / f"run_{run_idx}.train.log"
        eval_log = logs_dir / f"run_{run_idx}.eval.log"

        train_cmd = [args.python, "-m", "src.train", "--config", str(derived_path)]
        eval_cmd = [args.python, "-m", "src.eval", "--config", str(derived_path)]

        print(f"\n[run {run_idx}] overrides: {overrides_named}")
        print(f"  derived config: {derived_path}")
        print(f"  train: {' '.join(train_cmd)}  > {train_log}")
        print(f"  eval:  {' '.join(eval_cmd)}  > {eval_log}")

        if args.dry_run:
            results.append(
                {
                    "run_idx": run_idx,
                    "config_overrides": overrides_named,
                    "derived_config": str(derived_path),
                    "status": "planned",
                    "metrics": {},
                }
            )
            continue

        # Materialize the derived config now that we're actually running.
        dump_yaml(derived_path, derived)

        status = "ok"
        train_rc = run_subprocess(train_cmd, train_log)
        if train_rc != 0:
            status = f"train_failed(rc={train_rc})"
            print(f"  [run {run_idx}] train failed (rc={train_rc}); see {train_log}; continuing")
            results.append(
                {
                    "run_idx": run_idx,
                    "config_overrides": overrides_named,
                    "derived_config": str(derived_path),
                    "status": status,
                    "metrics": {},
                }
            )
            _write_results(out_dir, results)
            continue

        eval_rc = run_subprocess(eval_cmd, eval_log)
        if eval_rc != 0:
            status = f"eval_failed(rc={eval_rc})"
            print(f"  [run {run_idx}] eval failed (rc={eval_rc}); see {eval_log}; continuing")

        metrics_path = find_metrics_json(derived, run_dir)
        metrics = parse_metrics(metrics_path)
        if not metrics and status == "ok":
            status = "no_metrics"

        results.append(
            {
                "run_idx": run_idx,
                "config_overrides": overrides_named,
                "derived_config": str(derived_path),
                "metrics_path": str(metrics_path) if metrics_path else None,
                "status": status,
                "metrics": metrics,
            }
        )
        _write_results(out_dir, results)

    if not args.dry_run:
        _write_results(out_dir, results)

    print("\n" + render_markdown_table(results, sweep_keys))
    return 0


def _write_results(out_dir: Path, results: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
