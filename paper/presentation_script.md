# Presentation Script

**Distillation and Edge-Aware Loss for Efficient Monocular Depth Estimation**
ECE 4990 — Cal Poly Pomona — Spring 2026
Alexander Assal • Parsa Ghasemi

Target runtime: **~14 minutes** of speaking + 1 minute buffer for transitions.
Speaker labels: **[AA]** = Alexander Assal, **[PG]** = Parsa Ghasemi.
Times shown are cumulative.

---

## Slide 1 — Title  (0:00 – 0:20)
**[AA]**
Good afternoon. I'm Alexander Assal, and this is Parsa Ghasemi. Our project for ECE 4990 is *Distillation and Edge-Aware Loss for Efficient Monocular Depth Estimation*. The full source, paper, and slides are on GitHub at `arassal/depth-project`. We'll walk you through the problem, our method, the experiments, and what we learned.

---

## Slide 2 — Outline  (0:20 – 0:55)
**[PG]**
Here's the path we'll take. First the problem — why single-image depth estimation is hard. Then our method — we distill a large pretrained model into a small one, with an extra gradient supervision term. Then the datasets, NYU-v2 indoors and KITTI outdoors. Then our three-axis hyperparameter sweep — over the distillation weight α, the gradient weight β, and the learning rate. The main finding, in one sentence: **in the small-data regime, the learning rate dominates the loss balance.** I'll come back to that at the end.

---

## Slide 3 — Monocular depth is ill-posed  (0:55 – 1:35)
**[PG]**
Monocular depth estimation predicts a per-pixel depth map from a single RGB image. The catch is that a single image is consistent with infinitely many world geometries — without stereo, intrinsics, or active sensing, you can only recover depth up to an unknown scale and shift. Modern systems sidestep this by training for *relative* depth and aligning the scale at evaluation time.

The foundation model we build on is *Depth Anything* — Yang et al., CVPR 2024. It's a DPT depth head on a self-supervised DINOv2 backbone, trained on one-and-a-half million labeled images plus sixty-two million pseudo-labeled images. We use the Small variant (about 25 million parameters) as our **student** — it fits on a four-gigabyte laptop GPU — and the Large variant (about 335 million parameters) as our frozen **teacher**.

---

## Slide 4 — Related work  (1:35 – 2:15)
**[PG]**
Our backbone lineage starts with Eigen et al. 2014, the first deep monocular depth network. DPT — Ranftl 2021 — introduced the dense-prediction Transformer head we use. DINOv2 — Oquab 2023 — is the self-supervised encoder. MiDaS — Ranftl 2022 — introduced the relative-depth-plus-scale-alignment evaluation we follow. Depth Anything v1 and v2 are the pretrained checkpoints.

On the method side we adapt two pieces from the literature. Affine-invariant output distillation comes from Distill-Any-Depth, He et al. 2025. The multi-scale gradient L1 loss comes from ZoeDepth, Bhat et al. 2023. Our contribution is *not* a new architecture; it's an empirical study of how these two pieces behave when you fine-tune on a small dataset.

---

## Slide 5 — Architecture  (2:15 – 3:10)
**[AA]**
Here's our pipeline. The same RGB image feeds two networks in parallel. On the bottom is the **student** — Depth Anything Small, in blue — which is the only model with trainable weights. On top is the **teacher** — Depth Anything Large, in orange — loaded once and frozen, never updated.

Both produce a depth prediction. We compute three losses against those predictions. **L_base** compares the student to NYU-v2 ground truth. **L_distill** compares the student to the teacher's prediction. And **L_grad** is a multi-scale gradient loss against ground truth. They're combined into a total loss with weights α and β. Only the student's gradient flows back through the encoder.

The teacher exists purely as a dense, low-noise per-pixel supervision signal — it never sees gradient updates, so its behavior is completely deterministic.

---

## Slide 6 — Loss function  (3:10 – 4:00)
**[AA]**
The total loss has three terms with two mixing weights. (1 minus α) times the base loss, plus α times the distillation loss, plus β times the gradient loss.

Both the base and distillation losses are the same **affine-invariant L1**. We normalize each sample by its median and mean absolute deviation, *then* take the L1. The reason is that we're training relative depth — two predictions that agree on structure but disagree on overall scale should have *zero* loss. The median plus MAD normalization is robust to specular pixels that would explode a mean-and-standard-deviation version.

The gradient loss penalizes disagreement between predicted and ground-truth gradients at three downsampling scales — one, two, and four. So in the limits: α equals zero is pure supervised fine-tuning; α equals one is pure distillation; β controls boundary sharpness.

---

## Slide 7 — Training procedure  (4:00 – 4:35)
**[AA]**
Optimization is standard: AdamW with weight decay 1e-4, gradient clipping at L2 norm 1.0, mixed precision when CUDA is available. Batch size is 1 because we run on a four-gigabyte laptop GPU at 256×256.

On the right is the harness we wrote for the sweep. You give it a YAML grid, it takes the cartesian product, and it runs `train.py` and `eval.py` as subprocesses for each grid point. Each run writes into its own isolated artifact folder — that matters; we'll talk about why later. Everything aggregates into a single `results.json`.

---

## Slide 8 — Datasets and metrics  (4:35 – 5:15)
**[PG]**
We evaluate on two datasets. NYU-v2 — indoor scenes — split into 450 train, 60 validation, 59 test images, depth range 0 to 10 meters. And KITTI eigen-test — outdoor driving — 100 images, depth range 0 to 80 meters. Both are evaluated at 256×256 for internal comparability.

The metrics are the standard monocular depth ones, all computed after per-image least-squares scale and shift alignment. AbsRel — mean of absolute relative error — lower is better. RMSE — root mean squared error — lower is better. And the δ-thresholds — fraction of pixels where the prediction is within a factor of 1.25, 1.25 squared, or 1.25 cubed of ground truth — higher is better.

---

## Slide 9 — NYU-v2 test results  (5:15 – 6:15)
**[AA]**
Three rows tell our headline story. Top row: zero-shot Depth Anything Small — AbsRel 0.155, δ₁ 0.815, RMSE 0.54. That's our reference.

Middle row, in red: the **default** distillation configuration shipped in our repo — α=0.5, β=0.1, learning rate 1e-5. AbsRel almost doubles to 0.28. δ₁ drops to 0.63. RMSE jumps by 66%. The student collapsed.

Bottom row, in green: the **best configuration we found** after the sweep. α equals 0.7, β equals 0.3, but the key change is the learning rate — 1e-7, two orders of magnitude lower than the default. AbsRel drops to 0.1505, slightly *better* than zero-shot. δ₁ improves to 0.8164. Every single metric moves in the right direction.

The mean improvement is small, but it's consistent.

---

## Slide 10 — All five metrics, side by side  (6:15 – 6:50)
**[AA]**
The grouped bar chart drives the same point home visually. For every metric — AbsRel, RMSE, and all three δ-thresholds — the red bar (the default config) is worse than zero-shot. The green bar (our best config) matches or beats zero-shot on every one. That's the kind of evidence that tells you the improvement isn't an artifact of one metric; it shows up everywhere we look.

---

## Slide 11 — Training collapse  (6:50 – 7:35)
**[AA]**
Why does the default configuration fail so badly? Look at the training curves.

The left panel shows training loss decreasing normally over two epochs. Looks fine. The middle panel shows validation AbsRel — at epoch 1 it tracks zero-shot at 0.157. At epoch 2 it shoots up to 1.0. The right panel shows validation δ₁ — at epoch 1 it's around 0.82. At epoch 2 it drops to literally zero.

That signature — val AbsRel saturating at 1.0 and δ₁ at 0 — is the student emitting a near-constant prediction. The trigger is the combination of batch size 1, only 450 training images, and a learning rate of 1e-5. A few high-curvature mini-batches late in epoch 2 are enough to push the encoder into a degenerate region.

---

## Slide 12 — Loss component decomposition  (7:35 – 8:05)
**[AA]**
Here's how the three loss terms contribute. At epoch 1 the total per-batch loss is around 3.37 — base contributes about 1.36, distillation about 1.35, gradient about 0.66. At epoch 2 they all decrease together to about 2.59 total. The takeaway: the base and distillation losses are comparable in magnitude — they're both telling the student roughly the same story. The gradient term is smaller, as it should be — it's a boundary-sharpening signal, not the main objective.

---

## Slide 13 — Per-image distribution of the collapse  (8:05 – 8:40)
**[AA]**
To make the collapse concrete: on the left is a histogram of per-image AbsRel for zero-shot in blue versus the default distill in orange. The blue distribution is tight around 0.155. The orange distribution shifts right *and* grows a heavy tail past 0.4.

On the right is the same data as a scatter — zero-shot AbsRel on x, distilled AbsRel on y, dotted y-equals-x line. **Fifty-eight of fifty-nine test images get worse.** Only one is better. This isn't an outlier-driven story. The collapse is uniform across the test set.

---

## Slide 14 — Alpha sweep  (8:40 – 9:15)
**[PG]**
Now the ablations. First, the distillation weight α — that's the balance between ground truth and teacher supervision. We swept α from 0 to 1 in five steps, keeping β at 0.1 and learning rate at the default 1e-5.

The best point within this sweep is α equals 0.7 with AbsRel about 0.20 — but notice that **every single point is still worse than the zero-shot baseline of 0.155.** At this learning rate, no choice of α saves us. The shape of the curve has more to do with training stability than with the teacher-versus-ground-truth balance.

---

## Slide 15 — Learning rate, the dominant variable  (9:15 – 10:15)
**[PG]**
This is the headline slide. Same setup as before — fixed α at 0.7 — but now we sweep the learning rate over two orders of magnitude.

There's a sharp transition between 1e-6 and 5e-6. Below the transition — the green band — the recipe is stable; AbsRel sits right at the zero-shot baseline. Above the transition — the red band — the student collapses; AbsRel jumps from 0.151 to 0.186 and δ₁ drops from 0.815 to 0.749. The same band-width holds for δ₁ on the right panel.

This is *the* finding of the paper. **In our small-data regime, the learning rate doesn't just shift performance — it partitions the configuration space into "works" and "doesn't work."** The α and β weights only matter inside the stable band.

---

## Slide 16 — Beta sweep, second order  (10:15 – 10:40)
**[PG]**
For completeness, the gradient weight β. With α at 0.7 and learning rate at the stable 1e-7, we swept β across zero, 0.1, and 0.3. The improvement is monotonic but small — AbsRel goes from 0.152 to 0.150, δ₁ from 0.8142 to 0.8164. Once the learning rate is in the stable band, β is doing the secondary work it was designed for — boundary sharpening — and the magnitude of its effect matches that role.

---

## Slide 17 — Qualitative examples  (10:40 – 11:15)
**[PG]**
Four NYU-v2 test images, four panels each — RGB, ground truth depth, our prediction, and the pixelwise error. Two things to notice. First, the coarse scene structure is preserved on every example — the model knows what's near and what's far. Second, the error concentrates at depth discontinuities — object edges, doorways, transitions from foreground to background. That's exactly the regime the gradient loss is designed to address, which is why we added β in the first place.

---

## Slide 18 — KITTI cross-domain  (11:15 – 11:55)
**[PG]**
For out-of-distribution generalization, we evaluated on a 100-image KITTI eigen-test subset. The student was trained only on NYU-v2 indoors; KITTI is outdoor driving scenes. Both models — zero-shot and our distilled best — produce essentially the same AbsRel: 0.4022 versus 0.4035.

The scatter shows why: 31 images improve, 69 degrade slightly, but every point sits tightly along the y-equals-x diagonal. At a learning rate of 1e-7 the encoder *barely moves* from its pretrained initialization. The out-of-distribution behavior is inherited entirely from the original Depth Anything training mixture — our fine-tuning doesn't change it in either direction.

---

## Slide 19 — Discussion  (11:55 – 12:50)
**[AA]**
Two points from the sweeps. The variance from the α sweep is about 0.09 AbsRel; the variance from the lr sweep is about 0.03 AbsRel. So in raw magnitude α has a bigger spread. **But the lr range covers the boundary between "worse than zero-shot" and "better than zero-shot,"** while the α range mostly covers the difference between two collapse modes. That's what we mean when we say lr dominates.

Second — why is the gain at the stable end so small? Honestly: at 1e-7 the student barely moves from its pretrained weights. The ceiling of a recipe that doesn't move the encoder is, by construction, the pretrained model itself. To unlock more, you need a larger effective batch size — gradient accumulation across 16 or more steps — or the full cross-context distillation variant from He 2025, which we omitted for compute reasons; or simply more labeled data, so the ground truth becomes a more trustworthy signal.

---

## Slide 20 — Limitations  (12:50 – 13:25)
**[PG]**
Four things to call out. We evaluate at 256×256, which inflates absolute KITTI numbers relative to leaderboard evaluations — but our internal comparisons are valid since every row uses the same resolution. We only test cross-domain on one out-of-distribution dataset. We run one seed per grid point, so we don't have variance estimates — although the fact that the collapse pattern is reproducible across configurations suggests it's not a seed artifact. And we omit cross-context distillation, which is the second half of the Distill-Any-Depth recipe.

---

## Slide 21 — Conclusion  (13:25 – 14:10)
**[AA]**
To summarize. We built a clean, open-source distillation pipeline for Depth Anything Small — student plus frozen Large teacher plus multi-scale gradient loss. We ran a three-axis sweep across α, β, and learning rate on NYU-v2 and KITTI. Our empirically tuned configuration beats pretrained zero-shot on every NYU-v2 metric — AbsRel 0.1505 versus 0.1550, δ₁ 0.8164 versus 0.8146. The improvement does not transfer to KITTI at our conservative learning rate.

The headline finding is that in small-data distillation, **learning-rate sensitivity dominates loss balance** — the lr partitions the configuration space, and the loss weights only matter once you're in the stable region.

Everything is reproducible at `github.com/arassal/depth-project`. We're happy to take questions.

---

# Cheat sheet for Q&A

- **Why α=0.7 specifically?** Empirical, from the sweep. It's the maximum that doesn't get clobbered by the GT signal becoming noise at small N.
- **Why MAD and not std for normalization?** Specular highlights in NYU-v2 explode the mean; median is robust. Early runs with mean+std NaN'd in the first 100 steps.
- **What if you had more GPUs?** Larger effective batch via gradient accumulation, full cross-context distillation, more epochs. We expect the stable-zone window to widen and the headline gain to grow.
- **Why does the default config ship with lr=1e-5?** It was a generic fine-tuning default. Our finding is that it's the wrong default for this regime — and that's part of the contribution.
- **Where did the figures come from?** Generated by `src/make_training_figures.py` from the raw JSON outputs. Reproducible end-to-end.
- **What's the architecture novelty?** None. The novelty is empirical, not architectural — a controlled study of two existing pieces in the small-data regime where neither was previously evaluated.
- **Why no batch size > 1?** 4 GB GPU VRAM. We'd use gradient accumulation if we did it again.
