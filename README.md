# Depth Project

Monocular depth proof of concept built from the original `Depth Anything` presentation, then turned into a runnable repo with training, evaluation, plots, qualitative outputs, a web demo, and presentation assets.

## What This Repo Shows
- model: `LiheYoung/depth-anything-small-hf`
- task: monocular relative depth estimation
- indoor benchmark: `NYU-v2` proof-of-concept subset
- second dataset demo: `KITTI Tiny`
- outputs: metrics, plots, qualitative panels, presentation pages, `.pptx`, uploadable local web demo

This is a real result, not a mockup. The pretrained zero-shot model worked well as a relative-depth system. The small supervised fine-tune on a tiny local subset overfit and performed worse on the held-out test split.

## Final PoC Result

| Run | AbsRel | delta1 | Test images |
| --- | ---: | ---: | ---: |
| Zero-shot | `0.1467` | `0.8381` | `59` |
| Fine-tuned | `0.1810` | `0.7163` | `59` |

Interpretation:
- the system is doing real depth estimation
- the pretrained model is the strongest result in this repo
- the two-epoch fine-tune on `450` train images overfit the small split

## Dataset Sizes
- train: `450`
- val: `60`
- test: `59`

## Diagnostics Run
- split counts verified from manifest files
- metrics summaries verified against saved JSON outputs
- source tree compiled with `python -m compileall src`
- `DepthEstimationPipeline` loaded successfully from `LiheYoung/depth-anything-small-hf`

Full note: [docs/diagnostics.md](docs/diagnostics.md)

## What The Images Mean
The NYU-v2 qualitative panels are laid out as:
- `RGB`
- `GT` ground-truth depth
- `Pred` predicted depth
- `Error` aligned pixelwise error

These panels are best read as a relative-depth check:
- does the model separate foreground from background?
- does it preserve large scene layout?
- do object boundaries land in roughly the right place?

## Key Figures

### Training Curves
![Loss vs epoch](docs/assets/figures/loss_vs_epoch.png)
![Validation AbsRel vs epoch](docs/assets/figures/val_absrel_vs_epoch.png)
![Validation delta1 vs epoch](docs/assets/figures/val_delta1_vs_epoch.png)

### NYU-v2 Qualitative Example
Zero-shot:

![NYU zero-shot example](docs/assets/figures/nyu_zero_shot_000000.png)

Fine-tuned:

![NYU fine-tuned example](docs/assets/figures/nyu_finetuned_000000.png)

Second zero-shot example:

![NYU zero-shot example 2](docs/assets/figures/nyu_zero_shot_000004.png)

Second fine-tuned example:

![NYU fine-tuned example 2](docs/assets/figures/nyu_finetuned_000004.png)

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

Desktop package:
- `/home/alexander/Desktop/DepthProjectDemo`

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

This runs zero-shot inference on uploaded RGB images and saves outputs under `outputs/web_demo/`.

## Training Commands
Prepare the full NYU-v2 export:

```bash
cd /home/alexander/depth-project
source .venv/bin/activate
python src/prepare_nyu_v2.py --root /home/alexander/depth-project/data/nyu_v2 --val-count 512
```

Train:

```bash
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

## Honest Conclusion
This repo is a successful depth-estimation proof of concept, not a claim of successful task-specific fine-tuning. The zero-shot model is accurate enough to demonstrate meaningful depth structure. The custom fine-tuning recipe needs more data or better tuning before it should be presented as an improvement.
