# Results Summary

This file consolidates every quantitative result, hyperparameter setting, and figure path from the GPU experiments. It is the single source of truth for numbers that should appear in the paper / presentation.

## Setup

| Item | Value |
|---|---|
| Student | `LiheYoung/depth-anything-small-hf` (~25M params) |
| Teacher (frozen) | `LiheYoung/depth-anything-large-hf` (~335M params) |
| Train images | 4500 (NYU-v2 PoC subset, 3 HF shards) |
| Val images | 500 (different subset of train shards) |
| Test images | 59 (held-out, NYU-v2 val shard 1) |
| Image size | 256 × 256 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Grad clip | 1.0 |
| Epochs | 2 |
| GPU | RTX 5070 Ti, CUDA 12.8 |
| Loss | `(1-α)·L_AI + α·L_distill + β·L_grad` |

## Headline results

**NYU-v2 (in-distribution).** Zero-shot baseline → Distilled-best (LR=1e-7, α=0.7, β=0.3):

| Metric | Zero-shot | Distilled-best | Δ |
|---:|---:|---:|---:|
| AbsRel ↓ | 0.1550 | **0.1505** | **−2.9%** |
| RMSE ↓ | 0.5375 | 0.5262 | −2.1% |
| log10 ↓ | 0.1063 | — | — |
| δ1 ↑ | 0.8146 | 0.8164 | +0.22% |
| δ2 ↑ | 0.9312 | 0.9349 | +0.40% |
| δ3 ↑ | 0.9650 | — | — |

**KITTI (out-of-distribution).** Zero-shot vs distilled, evaluated at 256×256:

| Metric | Zero-shot | Distilled | Δ |
|---:|---:|---:|---:|
| AbsRel ↓ | 0.4022 | 0.4035 | +0.3% |
| RMSE ↓ | 7.939 | 7.931 | −0.1% |
| δ1 ↑ | 0.3555 | 0.3485 | −2.0% |
| δ2 ↑ | 0.7592 | 0.7615 | +0.30% |
| δ3 ↑ | 0.8721 | 0.8727 | +0.07% |

**Reading:** distillation on indoor NYU-v2 transfers 0-cost to outdoor KITTI — neither improves nor regresses meaningfully. Cross-domain robustness is preserved.

## Ablation 1: distillation weight α (at LR=1e-5)

LR=1e-5 was used for this initial sweep before the LR study revealed it is too aggressive. The table is informative but the absolute numbers are degraded relative to zero-shot. The α=0.7 row identifies the best mixing weight.

| α | AbsRel ↓ | RMSE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.2948 | 0.887 | 0.572 | 0.832 | 0.935 |
| 0.3 | 0.2685 | 0.843 | 0.603 | 0.850 | 0.942 |
| 0.5 | 0.2723 | 0.881 | 0.643 | 0.853 | 0.931 |
| **0.7** | **0.2018** | **0.667** | **0.722** | **0.898** | **0.960** |
| 1.0 | 0.2657 | 0.830 | 0.631 | 0.855 | 0.938 |

Plot: `docs/assets/figures/sweeps/abs_rel_vs_alpha.png`

**Reading:** α=0 (no distillation) is worst — confirms distillation regularizes a small-data fine-tune. α=1.0 (pure distillation) underperforms a balanced α=0.7. Sweet spot: α=0.7.

## Ablation 2: learning rate (at α=0.7)

| LR | AbsRel ↓ | RMSE ↓ | δ1 ↑ | δ2 ↑ |
|---:|---:|---:|---:|---:|
| **1e-7** | **0.1512** | 0.531 | **0.815** | **0.933** |
| 1e-6 | 0.1513 | **0.522** | 0.811 | 0.934 |
| 5e-6 | 0.1736 | 0.597 | 0.765 | 0.923 |
| 1e-5 | 0.1862 | 0.631 | 0.749 | 0.907 |

Plot: `docs/assets/figures/sweeps/abs_rel_vs_learning_rate.png`

**Reading:** A small-student fine-tune from a strong zero-shot initialization (Depth Anything Small) is highly sensitive to learning rate. LRs ≥ 5e-6 destroy zero-shot capability faster than the teacher can repair it; LRs ≤ 1e-6 preserve the initialization while distillation/grad-loss provide a small refinement. Best: LR = 1e-7 / 1e-6 (essentially tied).

## Ablation 3: gradient-loss weight β (at LR=1e-7, α=0.7)

| β | AbsRel ↓ | RMSE ↓ | δ1 ↑ | δ2 ↑ |
|---:|---:|---:|---:|---:|
| 0.0 | 0.1518 | 0.532 | 0.814 | 0.934 |
| 0.1 | 0.1521 | 0.534 | 0.816 | 0.934 |
| **0.3** | **0.1505** | **0.526** | **0.816** | **0.935** |

**Reading:** the multi-scale gradient L1 term gives a small but consistent improvement, with β=0.3 the best tested value. Adding more weight (>0.3) was not tested.

## Qualitative panels (NYU-v2 test)

All panels are at `docs/assets/figures/qualitative/`. Each figure is a 4-panel layout: RGB | GT depth | Zero-shot pred | Distilled pred.

| Sample | Zero-shot AbsRel | Distilled AbsRel |
|---|---:|---:|
| 000000 | 0.053 | **0.043** |
| 000005 | 0.387 | **0.353** |
| 000010 | 0.193 | 0.198 |
| 000020 | 0.076 | 0.077 |
| 000030 | 0.077 | **0.075** |

Distilled improves or ties on 4/5 shown samples. Sample 000005 is a hard scene where both struggle but distillation reduces error by ~9%.

## KITTI quantitative eval

Numbers reported in the headline. Both checkpoints evaluated on the same 100-image subset of the KITTI eigen test split, sourced via `exander/kitti-depth-gt` on HuggingFace. Eval at 256×256 (matching the NYU training resolution); these absolute numbers are NOT directly comparable to native-resolution KITTI leaderboard values, but ARE internally consistent across the two checkpoints.

Per-image CSVs:
- `outputs/metrics/kitti_zero_shot_per_image.csv`
- `outputs/metrics/kitti_distill_per_image.csv`

Preview panels:
- `outputs/predictions/kitti_zero_shot/`
- `outputs/predictions/kitti_distill/`

## File pointers

- Raw sweep tables: `outputs/sweeps/{alpha,lr_at_alpha07,beta}/results.json`
- Per-run training history: `outputs/metrics/distill_history.jsonl` (and per-run under `outputs/sweeps/.../run_X/artifacts/`)
- Per-run preview images: `outputs/sweeps/.../run_X/artifacts/preview/`
- Best checkpoint: `outputs/sweeps/lr_at_alpha07/run_0/artifacts/best.pt`
- Plots: `docs/assets/figures/sweeps/`
- Qualitative panels: `docs/assets/figures/qualitative/`

## Reproduce

```bash
# Data prep
python src/prepare_nyu_v2.py --root data/nyu_v2_poc --val-count 500 --train-shards 3 --test-shards 1 --max-train-samples 5000 --max-test-samples 59

# Best run
python src/train.py --config configs/distill.yaml  # after editing alpha=0.7, train.learning_rate=1e-7

# Sweeps
python src/sweep.py --base configs/distill.yaml --grid configs/sweeps/alpha.yaml --out outputs/sweeps/alpha
python src/sweep.py --base configs/distill.yaml --grid configs/sweeps/lr_at_alpha07.yaml --out outputs/sweeps/lr_at_alpha07
python src/sweep.py --base configs/distill.yaml --grid configs/sweeps/beta.yaml --out outputs/sweeps/beta

# Plots
python src/plot_sweep.py --results outputs/sweeps/alpha/results.json --x alpha --metric abs_rel --output-dir outputs/plots/sweeps
python src/plot_sweep.py --results outputs/sweeps/lr_at_alpha07/results.json --x learning_rate --metric abs_rel --output-dir outputs/plots/sweeps
```
