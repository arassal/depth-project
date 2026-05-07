# Diagnostics

This note records the repo-level checks run after the proof-of-concept outputs were generated.

## Runtime checks
- `python -m compileall src` completed successfully
- `DepthEstimationPipeline` loaded successfully from `LiheYoung/depth-anything-small-hf`
- `RefinedDepthModel` loaded successfully with a forward-pass smoke test at `256x256`

## Dataset checks
- `data/nyu_v2_poc/splits/train.txt`: `450` entries
- `data/nyu_v2_poc/splits/val.txt`: `60` entries
- `data/nyu_v2_poc/splits/test.txt`: `59` entries

## Metrics checks
- zero-shot teacher: `AbsRel 0.1467`, `delta1 0.8381`
- supervised-only student: `AbsRel 0.1810`, `delta1 0.7163`
- teacher-student student: `AbsRel 0.1682`, `delta1 0.7531`
- refined teacher-student student: `AbsRel 0.1591`, `delta1 0.7696`
- teacher-student vs supervised-only:
  - `AbsRel -0.0128`
  - `delta1 +0.0368`
- teacher-student vs teacher:
  - `AbsRel +0.0216`
  - `delta1 -0.0851`
- refined vs teacher-student:
  - `AbsRel -0.0092`
  - `delta1 +0.0165`
- refined vs zero-shot teacher:
  - `AbsRel +0.0124`
  - `delta1 -0.0685`

## Artifact checks
- qualitative NYU-v2 outputs exist for all `59` test images in:
  - `outputs/predictions/poc_zero_shot/`
  - `outputs/predictions/poc_finetuned/`
- qualitative teacher-student outputs exist for all `59` test images in:
  - `outputs/predictions/teacher_student_poc_test/`
- qualitative refined outputs exist for all `59` test images in:
  - `outputs/predictions/teacher_student_refined_test/`
- plots exist in `outputs/plots/poc/`
- teacher-student plots exist in `outputs/plots/teacher_student_poc/`
- refined architecture plots exist in `outputs/plots/teacher_student_refined/`
- teacher pseudo-label examples exist in:
  - `outputs/predictions/teacher_student_pseudo_samples/`
- KITTI Tiny zero-shot outputs exist in:
  - `/home/alexander/Desktop/DepthProjectDemo/KITTI_tiny_depth_outputs/image_02/`

## Interpretation
The project is technically complete as a proof of concept:
- the model runs
- the code compiles
- the data split is valid
- the metrics and plots are saved
- the qualitative outputs are available

The main technical finding is now sharper:
- the pretrained zero-shot teacher is still strongest
- the teacher-student student is better than the tiny supervised-only student
- the pseudo-label expansion helped, but it did not fully close the gap to the teacher
- the residual refinement head helped the student again, but still did not surpass the teacher
