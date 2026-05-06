# Slide-Driven Analysis

## What the Slides Actually Commit You To
The presentation is narrower than a full paper replication:

- It argues for `data scale` as the main novelty.
- It uses `Depth Anything Small` in the replication slide.
- It evaluates on `NYU-v2` using `AbsRel` and `delta1`.
- It explicitly warns that alignment, clipping, and resize policy can distort results.

That means the first executable version should not chase the full teacher-student paper pipeline. It should first make the replication slide true on your machine:

1. load `Depth Anything Small`
2. evaluate zero-shot on `NYU-v2`
3. fine-tune on `NYU-v2`
4. compare metrics and plots

Only then does pseudo-label self-training make sense.

## Engineering Translation of the Slides
### 1. Relative Depth, Not Pure Metric Depth
The slide deck repeatedly emphasizes scale ambiguity and affine-invariant fitting.

Practical consequence:
- training loss should compare normalized depth structure
- evaluation should align predictions before computing `AbsRel` and `delta1`

That is why this scaffold uses:
- robust normalization for loss
- scale-and-shift alignment for evaluation

### 2. Keep the Architecture Fixed
The slides say the architecture is not the novelty.

Practical consequence:
- do not write a custom encoder-decoder
- use the pretrained Hugging Face `Depth Anything` checkpoint
- spend effort on data handling and evaluation consistency instead

### 3. Match the Replication Constraints
Your own slide already states why results can differ:

- subset size
- CPU or weak hardware
- alignment mistakes
- clipping policy
- resize policy

Practical consequence:
- default image size is conservative
- default batch size is conservative
- metrics are standardized in one place
- plots are generated from the same logs every run

### 4. Self-Training Should Be Small First
The paper uses huge pseudo-labeled data. Your machine does not.

Practical consequence:
- use a small unlabeled set first
- use the fine-tuned baseline itself as the teacher
- stop if the self-training run does not beat the supervised baseline

## Machine Constraints
Current machine facts:

- GPU: `RTX 3050 4GB`
- `torch` and `transformers` are not installed in the current Python

This rules out:
- large-batch training
- large model variants as the default
- pretending the full pipeline is ready before dependency setup

It still supports:
- `Depth Anything Small`
- small-batch fine-tuning
- zero-shot evaluation
- pseudo-labeling on a modest unlabeled set

## Why This Execution Plan Is Correct
This setup preserves the presentation’s story:

1. zero-shot pretrained baseline
2. supervised NYU-v2 fine-tune
3. optional small-scale self-training

It also avoids the two main failure modes:

- overengineering a paper replica that will not fit the machine
- building a generic depth project that no longer matches the slides
