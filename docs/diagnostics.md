# Diagnostics

This note records the repo-level checks run after the proof-of-concept outputs were generated.

## Runtime checks
- `python -m compileall src` completed successfully
- `DepthEstimationPipeline` loaded successfully from `LiheYoung/depth-anything-small-hf`

## Dataset checks
- `data/nyu_v2_poc/splits/train.txt`: `450` entries
- `data/nyu_v2_poc/splits/val.txt`: `60` entries
- `data/nyu_v2_poc/splits/test.txt`: `59` entries

## Metrics checks
- zero-shot summary: `AbsRel 0.1467`, `delta1 0.8381`
- fine-tuned summary: `AbsRel 0.1810`, `delta1 0.7163`
- comparison conclusion: `fine_tuning_overfit_small_subset`

## Artifact checks
- qualitative NYU-v2 outputs exist for all `59` test images in:
  - `outputs/predictions/poc_zero_shot/`
  - `outputs/predictions/poc_finetuned/`
- plots exist in `outputs/plots/poc/`
- KITTI Tiny zero-shot outputs exist in:
  - `/home/alexander/Desktop/DepthProjectDemo/KITTI_tiny_depth_outputs/image_02/`

## Interpretation
The project is technically complete as a proof of concept:
- the model runs
- the code compiles
- the data split is valid
- the metrics and plots are saved
- the qualitative outputs are available

The main technical finding is unchanged: the pretrained zero-shot model is stronger than the tiny supervised fine-tune used in this local experiment.
