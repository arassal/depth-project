"""Regenerate all training-related figures from raw outputs/ data.

Reads:
  - outputs/metrics/distill_history.jsonl                 default-config training
  - outputs/sweeps/<sweep>/run_<i>/artifacts/history.jsonl per-run training
  - outputs/sweeps/<sweep>/results.json                    sweep summaries
  - outputs/metrics/{zero_shot,distill}_*_per_image.csv    per-image metrics

Writes to docs/assets/figures/training/.

Run from the repo root:
    python src/make_training_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
SWEEPS = OUT / "sweeps"
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figures" / "training"
FIGS.mkdir(parents=True, exist_ok=True)

# Consistent styling
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

INK = "#1f1a17"
ACCENT = "#A84425"
GOOD = "#2F7A46"
BAD = "#A83232"
MUTED = "#645A54"
ZS_COLOR = "#3b7dd8"
DT_COLOR = "#A84425"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_history(jsonl: Path) -> pd.DataFrame:
    return pd.DataFrame(load_jsonl(jsonl))


def save(fig: plt.Figure, name: str) -> None:
    path = FIGS / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ----------------------------------------------------------------------------
# 1. Loss-component decomposition of the default distill run
# ----------------------------------------------------------------------------
def fig_loss_components():
    df = load_history(METRICS / "distill_history.jsonl")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    epochs = df["epoch"].values
    base = df["train_base_loss"].values
    distill = df["train_distill_loss"].values
    grad = df["train_grad_loss"].values

    # Stacked bar chart per epoch
    bw = 0.55
    ax.bar(epochs, base, width=bw, color="#6c8ebf", label=r"$\mathcal{L}_{\mathrm{base}}$ (vs GT)")
    ax.bar(epochs, distill, width=bw, bottom=base, color="#f0b450", label=r"$\mathcal{L}_{\mathrm{distill}}$ (vs teacher)")
    ax.bar(epochs, grad, width=bw, bottom=base + distill, color="#c66565", label=r"$\mathcal{L}_{\mathrm{grad}}$ (multi-scale)")

    for ep, b, d, g in zip(epochs, base, distill, grad):
        ax.text(ep, b + d + g + 0.03, f"total\n{b+d+g:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK)

    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-batch loss (mean)")
    ax.set_title("Loss component decomposition  ($\\alpha{=}0.5$, $\\beta{=}0.1$, lr$=10^{-5}$)")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "loss_components.png")


# ----------------------------------------------------------------------------
# 2. Validation AbsRel/delta1 trajectories across all alpha sweep runs
# ----------------------------------------------------------------------------
def fig_all_runs_validation(sweep_name: str, hyperparam_label: str, output_name: str,
                            colors: list[str] | None = None):
    sweep_dir = SWEEPS / sweep_name
    results = json.loads((sweep_dir / "results.json").read_text())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    cmap = plt.get_cmap("plasma")
    n = len(results)
    for i, run in enumerate(results):
        run_idx = run["run_idx"]
        hist_path = sweep_dir / f"run_{run_idx}" / "artifacts" / "history.jsonl"
        if not hist_path.exists():
            continue
        df = load_history(hist_path)
        epochs = df["epoch"].values
        absrel = df["val_abs_rel"].values
        delta1 = df["val_delta1"].values
        if colors:
            color = colors[i]
        else:
            color = cmap(i / max(n - 1, 1))
        # Label uses the swept hyperparameter
        override_keys = list(run["config_overrides"].keys())
        # Use just the sweep variable for legend
        sweep_key = override_keys[0] if override_keys else "x"
        sweep_val = run["config_overrides"][sweep_key]
        label = f"{sweep_key}={sweep_val:g}" if isinstance(sweep_val, float) else f"{sweep_key}={sweep_val}"
        ax1.plot(epochs, absrel, "-o", color=color, label=label, markersize=4, linewidth=1.5)
        ax2.plot(epochs, delta1, "-o", color=color, label=label, markersize=4, linewidth=1.5)

    # Reference: zero-shot baseline
    zs = json.loads((METRICS / "zero_shot_nyu_summary.json").read_text())
    ax1.axhline(zs["abs_rel"], color="black", linestyle=":", linewidth=1.2, label="zero-shot baseline")
    ax2.axhline(zs["delta1"], color="black", linestyle=":", linewidth=1.2, label="zero-shot baseline")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation AbsRel  ($\\downarrow$)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Validation $\delta_{<1.25}$  ($\uparrow$)")
    ax1.set_title(f"Val AbsRel vs epoch — sweep over {hyperparam_label}")
    ax2.set_title(f"Val $\\delta_1$ vs epoch — sweep over {hyperparam_label}")
    ax1.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, output_name)


# ----------------------------------------------------------------------------
# 3. Per-image scatter — zero-shot vs distilled on NYU-v2
# ----------------------------------------------------------------------------
def fig_per_image_scatter():
    zs = pd.read_csv(METRICS / "zero_shot_nyu_per_image.csv")
    dt = pd.read_csv(METRICS / "distill_test_per_image.csv")
    merged = zs.merge(dt, on="sample_id", suffixes=("_zs", "_dt"))

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    improved = merged["abs_rel_dt"] < merged["abs_rel_zs"]
    ax.scatter(merged.loc[improved, "abs_rel_zs"], merged.loc[improved, "abs_rel_dt"],
               color=GOOD, s=22, alpha=0.85, label=f"distilled better ({improved.sum()})")
    ax.scatter(merged.loc[~improved, "abs_rel_zs"], merged.loc[~improved, "abs_rel_dt"],
               color=BAD, s=22, alpha=0.85, label=f"distilled worse ({(~improved).sum()})")
    lim_min = 0.0
    lim_max = max(merged["abs_rel_zs"].max(), merged["abs_rel_dt"].max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle=":", linewidth=1.0,
            label="y = x (no change)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Zero-shot AbsRel  ($\\downarrow$)")
    ax.set_ylabel("Distilled AbsRel  ($\\downarrow$)")
    ax.set_title("Per-image AbsRel — distilled (default) vs zero-shot  (NYU-v2)")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "per_image_scatter_default.png")


# ----------------------------------------------------------------------------
# 4. Histogram overlay — zero-shot vs distilled per-image AbsRel
# ----------------------------------------------------------------------------
def fig_histogram_comparison():
    zs = pd.read_csv(METRICS / "zero_shot_nyu_per_image.csv")
    dt = pd.read_csv(METRICS / "distill_test_per_image.csv")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bins = np.linspace(0, max(zs["abs_rel"].max(), dt["abs_rel"].max()) * 1.05, 25)
    ax.hist(zs["abs_rel"], bins=bins, alpha=0.55, color=ZS_COLOR, label=f"Zero-shot (mean {zs['abs_rel'].mean():.3f})", edgecolor="white")
    ax.hist(dt["abs_rel"], bins=bins, alpha=0.55, color=DT_COLOR, label=f"Distilled, default (mean {dt['abs_rel'].mean():.3f})", edgecolor="white")
    ax.axvline(zs["abs_rel"].mean(), color=ZS_COLOR, linestyle="--", linewidth=1.0)
    ax.axvline(dt["abs_rel"].mean(), color=DT_COLOR, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Per-image AbsRel  ($\\downarrow$)")
    ax.set_ylabel("Number of test images")
    ax.set_title("Per-image AbsRel distribution  (NYU-v2 test, 59 images)")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "absrel_histogram_comparison.png")


# ----------------------------------------------------------------------------
# 5. CDF of per-image AbsRel
# ----------------------------------------------------------------------------
def fig_cdf_comparison():
    zs = pd.read_csv(METRICS / "zero_shot_nyu_per_image.csv")
    dt = pd.read_csv(METRICS / "distill_test_per_image.csv")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for label, df, color in (("Zero-shot", zs, ZS_COLOR), ("Distilled (default)", dt, DT_COLOR)):
        x = np.sort(df["abs_rel"].values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=color, linewidth=2.0, label=label)
    ax.set_xlabel("AbsRel threshold")
    ax.set_ylabel("Fraction of test images with AbsRel $<$ threshold")
    ax.set_title("Cumulative AbsRel distribution  (NYU-v2)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    save(fig, "absrel_cdf.png")


# ----------------------------------------------------------------------------
# 6. Multi-metric grouped bar chart (3 settings × 5 metrics)
# ----------------------------------------------------------------------------
def fig_multimetric_bars():
    zs = json.loads((METRICS / "zero_shot_nyu_summary.json").read_text())
    dt = json.loads((METRICS / "distill_test_summary.json").read_text())
    # "Best" config from lr_at_alpha07 sweep (alpha=0.7, lr=1e-7, beta=0.1)
    # Use beta sweep run_2 (alpha=0.7, beta=0.3, lr=1e-7) for our best
    best = json.loads((SWEEPS / "beta" / "run_2" / "artifacts" / "metrics.json").read_text())

    metrics = ["abs_rel", "rmse", "delta1", "delta2", "delta3"]
    labels = ["AbsRel ↓", "RMSE ↓", "δ₁ ↑", "δ₂ ↑", "δ₃ ↑"]
    x = np.arange(len(metrics))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.bar(x - width, [zs[m] for m in metrics], width, color="#7a8ca8", label="Zero-shot")
    ax.bar(x,         [dt[m] for m in metrics], width, color=BAD,        label="Default distill (collapse)")
    ax.bar(x + width, [best[m] for m in metrics], width, color=GOOD,      label="Best swept (ours)")

    # Annotate values
    for i, m in enumerate(metrics):
        for offset, val in zip((-width, 0.0, width), (zs[m], dt[m], best[m])):
            ax.text(i + offset, val + 0.015, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.set_title("NYU-v2 test metrics — zero-shot vs default distill vs best swept")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="lightgray")
    ax.set_ylim(0, 1.18)
    save(fig, "multimetric_bars.png")


# ----------------------------------------------------------------------------
# 7. Combined sweep panel — alpha, lr, beta on one figure
# ----------------------------------------------------------------------------
def fig_sweep_panel():
    alpha = json.loads((SWEEPS / "alpha" / "results.json").read_text())
    lr = json.loads((SWEEPS / "lr_at_alpha07" / "results.json").read_text())
    beta = json.loads((SWEEPS / "beta" / "results.json").read_text())
    zs = json.loads((METRICS / "zero_shot_nyu_summary.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    def plot_sweep(ax, results, xkey, xlabel, log_x=False):
        xs = [r["config_overrides"][xkey] for r in results]
        absrel = [r["metrics"]["abs_rel"] for r in results]
        delta1 = [r["metrics"]["delta1"] for r in results]
        ax.plot(xs, absrel, "-o", color=ACCENT, label="AbsRel ($\\downarrow$)", linewidth=2, markersize=6)
        ax.axhline(zs["abs_rel"], color="black", linestyle=":", linewidth=1.2, label="zero-shot")
        # secondary y axis for delta1
        ax2 = ax.twinx()
        ax2.plot(xs, delta1, "-s", color=GOOD, label=r"$\delta_1$ ($\uparrow$)", linewidth=2, markersize=5)
        ax2.axhline(zs["delta1"], color="black", linestyle=":", linewidth=0.8, alpha=0.6)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", labelcolor=GOOD)
        ax2.set_ylabel(r"$\delta_1$", color=GOOD)
        ax.tick_params(axis="y", labelcolor=ACCENT)
        ax.set_ylabel("AbsRel", color=ACCENT)
        ax.set_xlabel(xlabel)
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        # Merge legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", frameon=False, fontsize=8)

    plot_sweep(axes[0], alpha, "alpha", r"$\alpha$ (distillation weight)")
    axes[0].set_title("Alpha sweep  (fixed lr$=10^{-5}$, $\\beta=0.1$)")

    plot_sweep(axes[1], lr, "learning_rate", "Learning rate (log scale)", log_x=True)
    axes[1].set_title(r"Learning-rate sweep  (fixed $\alpha=0.7$, $\beta=0.1$)")

    plot_sweep(axes[2], beta, "grad_weight", r"$\beta$ (gradient weight)")
    axes[2].set_title(r"Beta sweep  (fixed $\alpha=0.7$, lr$=10^{-7}$)")

    fig.tight_layout()
    save(fig, "sweep_panel.png")


# ----------------------------------------------------------------------------
# 8. KITTI per-image scatter
# ----------------------------------------------------------------------------
def fig_kitti_scatter():
    zs = pd.read_csv(METRICS / "kitti_zero_shot_per_image.csv")
    dt = pd.read_csv(METRICS / "kitti_distill_per_image.csv")
    merged = zs.merge(dt, on="sample_id", suffixes=("_zs", "_dt"))
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    improved = merged["abs_rel_dt"] < merged["abs_rel_zs"]
    ax.scatter(merged.loc[improved, "abs_rel_zs"], merged.loc[improved, "abs_rel_dt"],
               color=GOOD, s=22, alpha=0.85, label=f"distilled better ({improved.sum()})")
    ax.scatter(merged.loc[~improved, "abs_rel_zs"], merged.loc[~improved, "abs_rel_dt"],
               color=BAD, s=22, alpha=0.85, label=f"distilled worse ({(~improved).sum()})")
    lim_max = max(merged["abs_rel_zs"].max(), merged["abs_rel_dt"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], color="black", linestyle=":", linewidth=1.0,
            label="y = x")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("Zero-shot AbsRel")
    ax.set_ylabel("Distilled AbsRel")
    ax.set_title("Per-image AbsRel — KITTI eigen-test  (100 images)")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "kitti_per_image_scatter.png")


# ----------------------------------------------------------------------------
# 9. Lr vs final AbsRel — annotated story plot (the headline)
# ----------------------------------------------------------------------------
def fig_lr_headline():
    lr = json.loads((SWEEPS / "lr_at_alpha07" / "results.json").read_text())
    zs = json.loads((METRICS / "zero_shot_nyu_summary.json").read_text())
    xs = [r["config_overrides"]["learning_rate"] for r in lr]
    absrel = [r["metrics"]["abs_rel"] for r in lr]
    delta1 = [r["metrics"]["delta1"] for r in lr]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # Left: AbsRel
    ax1.plot(xs, absrel, "-o", color=ACCENT, linewidth=2.5, markersize=8, label="Distilled (ours)")
    ax1.axhline(zs["abs_rel"], color="black", linestyle="--", linewidth=1.4, label="Zero-shot baseline")
    ax1.set_xscale("log")
    ax1.set_xlabel("Learning rate")
    ax1.set_ylabel("AbsRel ($\\downarrow$ better)")
    # Shade the "collapse zone"
    ax1.axvspan(4e-6, 2e-5, color=BAD, alpha=0.08)
    ax1.text(1e-5, max(absrel) * 0.96, "collapse zone", color=BAD, fontsize=9, ha="center")
    # Shade the "stable zone"
    ax1.axvspan(8e-8, 2e-6, color=GOOD, alpha=0.08)
    ax1.text(3e-7, min(absrel) * 1.02, "stable zone\n(matches zero-shot)", color=GOOD, fontsize=9, ha="center")
    ax1.set_title("Headline: learning rate is the dominant variable")
    ax1.legend(loc="lower right", frameon=False)
    ax1.grid(True, which="both", alpha=0.25)

    # Right: delta1 mirror
    ax2.plot(xs, delta1, "-s", color=GOOD, linewidth=2.5, markersize=8, label="Distilled (ours)")
    ax2.axhline(zs["delta1"], color="black", linestyle="--", linewidth=1.4, label="Zero-shot baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("Learning rate")
    ax2.set_ylabel(r"$\delta_{<1.25}$ ($\uparrow$ better)")
    ax2.set_title(r"$\delta_1$ follows the same transition")
    ax2.legend(loc="lower left", frameon=False)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    save(fig, "lr_headline.png")


# ----------------------------------------------------------------------------
# 10. Loss vs val AbsRel scatter across all sweep runs (showing that training
#     loss is decoupled from validation quality in this regime)
# ----------------------------------------------------------------------------
def fig_loss_vs_val():
    rows = []
    for sweep_name in ("alpha", "lr_at_alpha07", "beta"):
        sd = SWEEPS / sweep_name
        results = json.loads((sd / "results.json").read_text())
        for run in results:
            run_idx = run["run_idx"]
            hist_path = sd / f"run_{run_idx}" / "artifacts" / "history.jsonl"
            if not hist_path.exists():
                continue
            hist = load_jsonl(hist_path)
            final = hist[-1]
            rows.append({
                "sweep": sweep_name,
                "train_loss": final["train_loss"],
                "val_abs_rel": final.get("val_abs_rel", np.nan),
                "alpha": run["config_overrides"].get("alpha"),
                "lr": run["config_overrides"].get("learning_rate"),
                "beta": run["config_overrides"].get("grad_weight"),
            })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = {"alpha": "#4477AA", "lr_at_alpha07": "#EE6677", "beta": "#228833"}
    for sweep_name, color in colors.items():
        sub = df[df["sweep"] == sweep_name]
        ax.scatter(sub["train_loss"], sub["val_abs_rel"], color=color, s=60,
                   label=sweep_name.replace("_at_alpha07", " (lr at $\\alpha=0.7$)"),
                   edgecolor="white", linewidth=1)
    # Annotate the headline point — best stable run
    best_idx = df["val_abs_rel"].idxmin()
    ax.annotate(f"best stable\n(lr={df.loc[best_idx, 'lr']:.0e})",
                xy=(df.loc[best_idx, "train_loss"], df.loc[best_idx, "val_abs_rel"]),
                xytext=(df.loc[best_idx, "train_loss"] + 0.05,
                        df.loc[best_idx, "val_abs_rel"] - 0.05),
                fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_xlabel("Final epoch train loss")
    ax.set_ylabel("Final val AbsRel")
    ax.set_title("Train loss vs val AbsRel — all 12 sweep runs")
    ax.legend(loc="upper left", frameon=False, title="sweep")
    save(fig, "train_loss_vs_val_absrel.png")


# ----------------------------------------------------------------------------
# 11. Single training-curves grid (3-panel: train loss / val AbsRel / val delta1)
#     for default distill — kept for the slides but with consistent new styling
# ----------------------------------------------------------------------------
def fig_training_curves_default():
    df = load_history(METRICS / "distill_history.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "-o", color=ACCENT, linewidth=2.5, markersize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("Training loss")
    ax.set_xticks(df["epoch"])

    ax = axes[1]
    ax.plot(df["epoch"], df["val_abs_rel"], "-o", color=ACCENT, linewidth=2.5, markersize=8,
            label="Distilled")
    ax.axhline(0.155, color="black", linestyle=":", linewidth=1.2, label="Zero-shot")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AbsRel")
    ax.set_title("Val AbsRel — collapses at epoch 2")
    ax.set_xticks(df["epoch"])
    ax.legend(loc="best", frameon=False)
    # Annotation
    ax.annotate("collapse", xy=(2, df["val_abs_rel"].iloc[-1]),
                xytext=(1.45, 0.7), fontsize=10, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1))

    ax = axes[2]
    ax.plot(df["epoch"], df["val_delta1"], "-o", color=GOOD, linewidth=2.5, markersize=8,
            label="Distilled")
    ax.axhline(0.815, color="black", linestyle=":", linewidth=1.2, label="Zero-shot")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Val $\delta_{<1.25}$")
    ax.set_title(r"Val $\delta_1$ — drops to 0 at epoch 2")
    ax.set_xticks(df["epoch"])
    ax.legend(loc="best", frameon=False)
    ax.annotate("collapse", xy=(2, df["val_delta1"].iloc[-1]),
                xytext=(1.45, 0.3), fontsize=10, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD, lw=1))

    fig.suptitle("Default distill config training dynamics  ($\\alpha=0.5$, $\\beta=0.1$, lr$=10^{-5}$)",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    save(fig, "training_curves_default.png")


if __name__ == "__main__":
    print(f"Writing figures to {FIGS.relative_to(ROOT)}/")
    fig_loss_components()
    fig_training_curves_default()
    fig_all_runs_validation("alpha", r"$\alpha$", "val_curves_alpha.png")
    fig_all_runs_validation("lr_at_alpha07", "learning rate", "val_curves_lr.png")
    fig_all_runs_validation("beta", r"$\beta$", "val_curves_beta.png")
    fig_per_image_scatter()
    fig_histogram_comparison()
    fig_cdf_comparison()
    fig_multimetric_bars()
    fig_sweep_panel()
    fig_lr_headline()
    fig_kitti_scatter()
    fig_loss_vs_val()
    print("done.")
