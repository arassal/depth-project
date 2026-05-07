# Multi-Dataset Workflow

This project started with `NYU-v2` as the first labeled dataset. The clean next step is to keep `NYU-v2` as the evaluation anchor and progressively add more labeled training data through merged manifests.

## Recommended order
1. `NYU-v2`
2. `KITTI` or another outdoor depth dataset with aligned image/depth pairs
3. optional pseudo-labeled unlabeled images

## Why this approach
- no architecture rewrite
- no separate trainer per dataset
- no custom sampler unless it becomes necessary later
- validation and test stay stable on the anchor dataset

## How it works
The data loader already accepts absolute paths. That means one merged manifest can point at samples from multiple dataset roots.

## Build a merged train manifest
Example:

```bash
cd /home/alexander/depth-project
source .venv/bin/activate
python src/merge_manifests.py \
  --absolute-paths \
  --manifest /home/alexander/depth-project/data/nyu_v2_poc/splits/train.txt \
  --manifest /path/to/second_dataset/splits/train.txt \
  --output /home/alexander/depth-project/data/manifests/train_multi.txt
```

## Train with the merged manifest

```bash
python src/train.py --config configs/poc_multi_dataset_template.yaml
python src/eval.py --config configs/poc_multi_dataset_template.yaml \
  --checkpoint /home/alexander/depth-project/checkpoints/poc_multi_dataset_best.pt
```

## Current state
- first dataset already used: `NYU-v2` PoC subset
- second dataset already used for zero-shot inference demo: `KITTI Tiny`
- second labeled training dataset is not wired in yet because there is no local KITTI depth-ground-truth split present

## Practical next addition
If you add a labeled `KITTI` depth split locally, this repo is now ready to:
- generate a KITTI manifest
- merge it with the existing NYU manifest
- train one combined model without changing the architecture
