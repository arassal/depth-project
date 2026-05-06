# Depth Anything Demonstration Report

Author: Alexander Assal  
Date: May 6, 2026  
Project repo: `https://github.com/arassal/depth-project`

## 1. Purpose
This report documents the complete implementation project built from the presentation:

`Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data`

The goal was not to re-create the full 63.5M-image paper training pipeline. The goal was to build a clean, reproducible monocular depth project that matches the presentation’s core claims:

- monocular depth estimation from a single RGB image
- use of a pretrained `Depth Anything` model
- evaluation on `NYU-v2`
- paper-aligned metrics: `AbsRel` and `delta1`
- a path toward self-training with pseudo-labels

This report covers:

- the original presentation story
- the engineering translation of that story
- the implemented codebase
- dataset preparation
- evaluation and plotting workflow
- GitHub publication
- current project status

## 2. Original Presentation Summary
The presentation explains the paper’s central idea:

- depth estimation improves dramatically when training data scales up
- the key novelty is not a new architecture
- the key novelty is training with massive pseudo-labeled unlabeled image data

The deck emphasizes five technical points:

1. Monocular depth estimation is ill-posed.
2. Absolute metric depth is ambiguous from one image.
3. `Depth Anything` uses a pretrained visual encoder plus a depth head.
4. The model is trained with relative-depth reasoning rather than strict metric matching.
5. Evaluation depends heavily on correct alignment, clipping, and preprocessing.

The presentation’s replication slide narrows the practical project scope even further:

- use `Depth Anything Small`
- use `NYU-v2`
- compute `AbsRel`
- compute `delta1`
- be careful with alignment and resize choices

That slide is the reason this implementation focuses on a lean, defensible replication instead of an oversized research pipeline.

## 3. Project Design Decisions
The project was designed to match both the presentation and the available hardware.

### Chosen dataset
`NYU-v2`

Reason:

- directly matches the presentation
- standard benchmark for indoor monocular depth
- realistic for a local implementation
- good fit for paper-style replication

### Chosen model
`LiheYoung/depth-anything-small-hf`

Reason:

- matches the presentation’s replication direction
- lighter than Base or Large
- realistic on an `RTX 3050 4GB`

### Chosen training objective
Affine-invariant relative depth fitting

Reason:

- consistent with the presentation’s emphasis on scale ambiguity
- avoids pretending we are doing strict metric-depth recovery
- better aligned with the paper’s stated logic

### Chosen evaluation metrics
- `AbsRel`
- `delta1`

Reason:

- these are explicitly called out in the presentation
- they are standard and easy to compare across runs

## 4. Implemented Project Structure
The project lives at:

`/home/alexander/depth-project`

GitHub repo:

`https://github.com/arassal/depth-project`

Key project files:

- `README.md`
- `requirements.txt`
- `configs/baseline.yaml`
- `configs/self_training.yaml`
- `src/train.py`
- `src/eval.py`
- `src/plots.py`
- `src/pseudo_label.py`
- `src/prepare_nyu_v2.py`
- `src/make_manifest.py`
- `src/slide_analysis.md`

### What each component does
`train.py`

- fine-tunes `Depth Anything Small`
- logs training loss and validation metrics
- saves the best checkpoint by validation `AbsRel`

`eval.py`

- loads a trained checkpoint
- evaluates on the test split
- computes aligned `AbsRel` and `delta1`
- saves per-image metrics
- saves qualitative preview panels

`plots.py`

- reads logged metrics
- generates report-ready plots

`pseudo_label.py`

- runs the trained baseline as a teacher
- generates pseudo-depth maps for unlabeled RGB images

`prepare_nyu_v2.py`

- downloads the practical NYU-v2 shard source from Hugging Face
- decodes the original `.h5` samples
- exports train, val, and test folders plus manifests

## 5. Environment and Hardware
Local hardware detected on May 6, 2026:

- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`
- VRAM: `4GB`
- Python: `3.12.3`

The project virtual environment was created at:

`/home/alexander/depth-project/.venv`

Installed core packages include:

- `torch`
- `torchvision`
- `transformers`
- `datasets`
- `h5py`
- `numpy`
- `matplotlib`
- `pandas`
- `pyyaml`

This setup is sufficient for:

- zero-shot evaluation
- small-batch fine-tuning
- plot generation
- pseudo-label generation on modest data sizes

## 6. Dataset Preparation
The most appropriate dataset for this project is `NYU-v2`.

### Why not the full paper data recipe?
The original paper uses:

- 1.5M labeled images
- 62M pseudo-labeled images

That full setup is not the right first implementation for this machine or this presentation replication. The project therefore uses:

- `NYU-v2` for the supervised baseline
- optional small-scale unlabeled data for later pseudo-label self-training

### Dataset source chosen
Practical source used:

`sayakpaul/nyu_depth_v2` on Hugging Face

This source packages the data as tar shards containing `.h5` samples with RGB and depth.

### Dataset export workflow
The project now includes:

```bash
python src/prepare_nyu_v2.py --root /home/alexander/depth-project/data/nyu_v2 --val-count 512
```

That command:

- downloads the dataset shards
- decodes RGB and depth
- writes PNG RGB images
- writes `.npy` depth arrays
- creates:
  - `splits/train.txt`
  - `splits/val.txt`
  - `splits/test.txt`

## 7. Training and Evaluation Workflow
The baseline workflow is:

1. Prepare `NYU-v2`
2. Train the baseline model
3. Evaluate on the test set
4. Generate plots
5. Optionally run pseudo-label self-training

### Baseline training command
```bash
cd /home/alexander/depth-project
source .venv/bin/activate
python src/train.py --config configs/baseline.yaml
```

### Baseline evaluation command
```bash
python src/eval.py --config configs/baseline.yaml --checkpoint checkpoints/baseline_best.pt
```

### Plot generation command
```bash
python src/plots.py \
  --metrics-log outputs/metrics/baseline_history.jsonl \
  --per-image outputs/metrics/baseline_test_per_image.csv \
  --output-dir outputs/plots/baseline
```

### Expected report plots
- training loss vs epoch
- validation loss vs epoch
- validation `AbsRel` vs epoch
- validation `delta1` vs epoch
- test per-image `AbsRel` histogram
- qualitative RGB / GT / prediction / error panels

## 8. Testing and Demonstration Assets
A standalone testing runbook was created at:

`/home/alexander/Desktop/depth_project_testing.md`

It includes:

- environment activation
- syntax checks
- import checks
- pretrained checkpoint load verification
- dataset preparation commands
- training commands
- evaluation commands
- plot generation commands
- self-training commands

This makes the project demonstrable even without relying on memory or ad hoc terminal notes.

## 9. GitHub Setup
The project was fully initialized as a Git repository and published.

Repository:

`https://github.com/arassal/depth-project`

Status:

- private GitHub repository
- default branch: `main`
- local repo initialized and pushed successfully

Commits currently include:

- `Initialize depth training workspace`
- `Add NYU-v2 preparation workflow`

## 10. Current Status on May 6, 2026
This is the honest project status at the time of this report.

### Completed
- presentation analyzed
- project plan created
- training workspace scaffolded
- configs created
- training script implemented
- evaluation script implemented
- plotting script implemented
- pseudo-label script implemented
- NYU-v2 preparation script implemented
- repo created and pushed to GitHub
- testing runbook written to the Desktop
- dependencies installed and verified
- pretrained `Depth Anything Small` checkpoint successfully loaded

### In progress
- full `NYU-v2` dataset download and export

At the time of reporting, the dataset process is actively running:

```bash
python /home/alexander/depth-project/src/prepare_nyu_v2.py \
  --root /home/alexander/depth-project/data/nyu_v2 \
  --val-count 512
```

This means the project infrastructure is complete, but baseline training metrics and plots are not yet available because the dataset export must finish first.

## 11. What the Finished Demonstration Will Show
Once dataset export completes and the baseline is trained, the finished demo will show the full end-to-end story from the presentation:

### Part 1. Problem framing
- monocular depth from a single image
- why scale ambiguity matters
- why relative depth is the right framing

### Part 2. Model choice
- `Depth Anything Small`
- pretrained vision backbone
- depth estimation head

### Part 3. Dataset
- `NYU-v2` as the supervised benchmark
- train / val / test split export
- manifest-based loading

### Part 4. Training
- fine-tuning on `NYU-v2`
- affine-invariant loss
- validation logging

### Part 5. Evaluation
- aligned `AbsRel`
- aligned `delta1`
- per-image metric breakdown

### Part 6. Visualization
- training curves
- metric curves
- qualitative prediction panels
- histogram of test difficulty / error spread

### Part 7. Extension
- pseudo-label generation
- self-training run
- comparison against the baseline

## 12. Limitations
This project intentionally does not claim the following:

- full paper-scale training on 63.5M images
- exact reproduction of the CVPR 2024 published numbers
- large-model training
- metric-depth recovery from a single RGB image without caveats

These are not failures. They are scope decisions made to keep the project technically honest and reproducible.

## 13. Conclusion
The project successfully transforms the original presentation into a real engineering artifact:

- a published GitHub repository
- a local runnable training environment
- dataset-preparation tooling
- training, evaluation, and plotting code
- a testing runbook
- a path from supervised baseline to self-training

The only remaining runtime step before a fully populated results section is completion of the `NYU-v2` export, followed by the first baseline training run.

At that point, the project will have the complete demonstration flow needed for the original presentation:

- theory
- implementation
- dataset
- training
- metrics
- plots
- qualitative outputs
- extension path
