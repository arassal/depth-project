# Depth Project

Lean monocular depth training workspace based on the `Depth Anything` presentation.

## What This Implements
- `Depth Anything Small` fine-tuning
- `NYU-v2`-style supervised training
- `AbsRel` and `delta1` evaluation
- paper-aligned qualitative plots
- optional pseudo-label generation for self-training

## Why This Is Minimal
- one labeled dataset format
- one baseline model
- one loss family
- one evaluation protocol
- one optional self-training step

## Expected Data Layout
```text
data/
  nyu_v2/
    images/
    depth/
    splits/
      train.txt
      val.txt
      test.txt
  unlabeled/
```

Each split file should contain:
```text
relative/path/to/image.png relative/path/to/depth.png
```

Depth paths are resolved relative to `data/nyu_v2/`.

## Environment
Create a virtual environment and install the requirements:

```bash
cd /home/alexander/depth-project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU training, install a CUDA-compatible PyTorch build first, then run:

```bash
pip install -r requirements.txt --no-deps
```

## Baseline Run
Prepare `NYU-v2` from the Hugging Face shard source:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_nyu_v2.py --root /home/alexander/depth-project/data/nyu_v2 --val-count 512
```

Train:

```bash
source .venv/bin/activate
python src/train.py --config configs/baseline.yaml
```

Evaluate:

```bash
python src/eval.py --config configs/baseline.yaml --checkpoint checkpoints/baseline_best.pt
```

Generate plots:

```bash
python src/plots.py --metrics-log outputs/metrics/baseline_history.jsonl --per-image outputs/metrics/baseline_test_per_image.csv --output-dir outputs/plots/baseline
```

## Self-Training Run
Generate pseudo-labels:

```bash
python src/pseudo_label.py --config configs/self_training.yaml --checkpoint checkpoints/baseline_best.pt
```

Then train with:

```bash
python src/train.py --config configs/self_training.yaml
```

## Notes
- This project assumes relative depth evaluation. It does not try to force perfect metric-depth replication of the full paper.
- The defaults are chosen for your `RTX 3050 4GB`, so batch size and image size are conservative.
- If `train.txt` and `val.txt` do not exist yet, create them before training.
