# Depth Project

Monocular depth proof of concept built from the original `Depth Anything` presentation, then extended into a closer paper-style teacher-student workflow.

## What This Repo Shows
- model: `LiheYoung/depth-anything-small-hf`
- architecture: `Depth Anything Small` / DPT-style monocular depth estimation
- benchmark: `NYU-v2` proof-of-concept subset
- teacher-student expansion: `450` labeled indoor images plus `2000` teacher-pseudo-labeled indoor images
- outputs: metrics, plots, qualitative panels, comparison galleries, HTML presentation, `.pptx`, local upload demo

## Final Story
This repo now shows three stages instead of just one:
1. zero-shot teacher
2. small supervised-only student
3. teacher-student student trained on labeled plus pseudo-labeled data

That is much closer to the original paper’s story than the first PoC version.

## Main Results

| Run | Training data | AbsRel | delta1 | Test images |
| --- | --- | ---: | ---: | ---: |
| Zero-shot teacher | pretrained only | `0.1467` | `0.8381` | `59` |
| Supervised-only student | `450` labeled | `0.1810` | `0.7163` | `59` |
| Teacher-student run | `450` labeled + `2000` pseudo-labeled | `0.1682` | `0.7531` | `59` |

Interpretation:
- the zero-shot teacher is still the best overall result
- the teacher-student run improved over the small supervised-only student
- the added pseudo-labeled data helped, but not enough to beat the teacher

## Data Growth
- labeled seed set: `450`
- validation set: `60`
- held-out test set: `59`
- pseudo-labeled indoor expansion: `2000`
- total teacher-student training set: `2450`

![Teacher-student data growth](docs/assets/figures/data_growth.png)

## Why This Matches The Paper Better
- architecture stayed the same
- data scale increased through pseudo-labeling rather than architecture changes
- the student was trained on more data than the original labeled seed set
- the repo now explicitly shows `teacher -> pseudo labels -> student retraining`

## What The Images Mean
The NYU-v2 qualitative panels are laid out as:
- `RGB`
- `GT` ground-truth depth
- `Pred` predicted depth
- `Error` aligned pixelwise error

The teacher pseudo-label panels are laid out as:
- `Unlabeled RGB`
- `Teacher pseudo-depth`

## Key Figures

### Three-Run Comparison
![Run comparison](docs/assets/figures/run_comparison.png)

### Teacher-Student Training Curves
![Teacher-student loss](docs/assets/figures/teacher_student_loss_vs_epoch.png)
![Teacher-student validation AbsRel](docs/assets/figures/teacher_student_val_absrel_vs_epoch.png)
![Teacher-student validation delta1](docs/assets/figures/teacher_student_val_delta1_vs_epoch.png)

### NYU-v2 Sample 000000
Zero-shot teacher:

![Zero-shot sample](docs/assets/figures/nyu_zero_shot_000000.png)

Supervised-only student:

![Supervised-only sample](docs/assets/figures/nyu_finetuned_000000.png)

Teacher-student run:

![Teacher-student sample](docs/assets/figures/teacher_student_000000.png)

### Teacher Pseudo-Label Examples
![Teacher pseudo sample 1](docs/assets/figures/teacher_pseudo_00.png)
![Teacher pseudo sample 2](docs/assets/figures/teacher_pseudo_01.png)

### KITTI Tiny Demo Example
Input frame:

![KITTI input](docs/assets/figures/kitti_input_0000000000.png)

Predicted depth:

![KITTI depth](docs/assets/figures/kitti_depth_0000000000.png)

## Main Artifacts
- results page: [docs/results.html](docs/results.html)
- presentation page: [docs/presentation.html](docs/presentation.html)
- pptx: [docs/DepthProjectPresentation.pptx](docs/DepthProjectPresentation.pptx)
- diagnostics note: [docs/diagnostics.md](docs/diagnostics.md)
- multi-dataset note: [docs/multi_dataset_workflow.md](docs/multi_dataset_workflow.md)

Desktop package:
- `/home/alexander/Desktop/DepthProjectDemo`

## Teacher-Student Commands
Export the indoor pseudo-label pool:

```bash
cd /home/alexander/depth-project
source .venv/bin/activate
python src/prepare_indoor_pseudo_pool.py \
  --root /home/alexander/depth-project/data/unlabeled/indoor_teacher_pool \
  --limit 2000
```

Generate pseudo labels with the teacher:

```bash
python src/pseudo_label.py --config configs/teacher_student_poc.yaml
```

Train the student:

```bash
python src/train.py --config configs/teacher_student_poc.yaml
```

Evaluate the student:

```bash
python src/eval.py --config configs/teacher_student_poc.yaml \
  --checkpoint checkpoints/teacher_student_poc_best.pt
```

## Local Web Demo
Run:

```bash
cd /home/alexander/depth-project
source .venv/bin/activate
python src/demo_web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

This runs the pretrained zero-shot teacher on uploaded RGB images and saves outputs under `outputs/web_demo/`.

## Honest Conclusion
The project now matches the original paper’s logic much more closely: start with a strong teacher, expand the dataset with pseudo labels, and train a student on more data. In this local run, the student improved over the small supervised-only baseline but still did not surpass the pretrained teacher.
