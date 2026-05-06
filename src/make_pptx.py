from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path("/home/alexander/depth-project")
OUT = ROOT / "outputs"
DESKTOP = Path("/home/alexander/Desktop/DepthProjectDemo")

METRICS = OUT / "metrics"
PLOTS = OUT / "plots" / "poc"
PREDS = OUT / "predictions"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


zero = load_json(METRICS / "poc_zero_shot_summary.json")
fine = load_json(METRICS / "poc_finetuned_summary.json")
comp = load_json(METRICS / "poc_comparison.json")


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


BG = RGBColor(247, 240, 229)
INK = RGBColor(31, 26, 23)
MUTED = RGBColor(100, 90, 84)
ACCENT = RGBColor(168, 68, 37)


def set_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG


def add_title(slide, title: str, subtitle: str | None = None):
    tx = slide.shapes.add_textbox(Inches(0.55), Inches(0.3), Inches(12.2), Inches(1.0))
    p = tx.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = INK
    if subtitle:
        p2 = tx.text_frame.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(13)
        r2.font.color.rgb = MUTED


def add_bullets(slide, items: list[str], left: float, top: float, width: float, height: float, font_size: int = 20):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.size = Pt(font_size)
        p.font.color.rgb = INK
        p.space_after = Pt(8)


def add_metric_box(slide, title: str, value: str, left: float, top: float):
    shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(2.2), Inches(1.25))
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(255, 250, 241)
    shape.line.color.rgb = RGBColor(220, 205, 192)

    tf = shape.text_frame
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(13)
    p1.font.color.rgb = MUTED
    p2 = tf.add_paragraph()
    p2.text = value
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT


def add_caption(slide, text: str, left: float, top: float, width: float):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.5))
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(12)
    p.font.color.rgb = MUTED


# Slide 1
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Depth Anything Proof of Concept", "Alexander Assal • GitHub: https://github.com/arassal/depth-project")
add_bullets(
    slide,
    [
        "Goal: build a clean monocular depth proof of concept from the original Depth Anything presentation",
        "Model: Depth Anything Small",
        "Dataset: NYU-v2 subset",
        "Outputs: metrics, plots, before/after image panels, interactive upload demo",
    ],
    0.8,
    1.5,
    11.6,
    3.0,
)

# Slide 2
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "What Model Was Used")
add_bullets(
    slide,
    [
        "Pretrained model: LiheYoung/depth-anything-small-hf",
        "Architecture family: Depth Anything / DPT-style monocular depth estimation",
        "Strength: strong zero-shot relative depth structure from a single RGB image",
        "Limitation: not guaranteed metric-accurate real-world distance from one image alone",
    ],
    0.8,
    1.2,
    6.1,
    4.0,
)
add_bullets(
    slide,
    [
        "This system is best interpreted as a relative-depth model",
        "It can tell what is nearer and farther",
        "It should not be sold as exact distance measurement without calibration",
    ],
    7.0,
    1.6,
    5.2,
    3.0,
    font_size=18,
)

# Slide 3
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Dataset and Training Setup")
add_metric_box(slide, "Train", "450", 0.8, 1.2)
add_metric_box(slide, "Val", "60", 3.25, 1.2)
add_metric_box(slide, "Test", "59", 5.7, 1.2)
add_metric_box(slide, "Epochs", "2", 8.15, 1.2)
add_bullets(
    slide,
    [
        "Image size: 256x256",
        "Batch size: 1",
        "Optimizer: AdamW",
        "Learning rate: 1e-5",
        "Loss: affine-invariant relative depth loss",
        "Evaluation metrics: AbsRel and delta1",
    ],
    0.8,
    3.0,
    5.0,
    3.4,
)
add_bullets(
    slide,
    [
        "Zero-shot baseline was measured before fine-tuning",
        "Fine-tuned checkpoint was selected using validation AbsRel",
        "This was intentionally a small PoC run, not a full paper-scale replication",
    ],
    6.3,
    3.0,
    5.8,
    3.4,
    font_size=18,
)

# Slide 4
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Training Outcome")
slide.shapes.add_picture(str(PLOTS / "loss_vs_epoch.png"), Inches(0.7), Inches(1.3), width=Inches(4.0))
slide.shapes.add_picture(str(PLOTS / "val_absrel_vs_epoch.png"), Inches(4.75), Inches(1.3), width=Inches(4.0))
slide.shapes.add_picture(str(PLOTS / "val_delta1_vs_epoch.png"), Inches(8.8), Inches(1.3), width=Inches(4.0))
add_caption(slide, "Loss vs epoch", 0.7, 5.4, 4.0)
add_caption(slide, "Validation AbsRel vs epoch", 4.75, 5.4, 4.0)
add_caption(slide, "Validation delta1 vs epoch", 8.8, 5.4, 4.0)

# Slide 5
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Test Results")
add_metric_box(slide, "Zero-shot AbsRel", f"{zero['abs_rel']:.4f}", 0.8, 1.2)
add_metric_box(slide, "Zero-shot delta1", f"{zero['delta1']:.4f}", 3.25, 1.2)
add_metric_box(slide, "Fine-tuned AbsRel", f"{fine['abs_rel']:.4f}", 5.7, 1.2)
add_metric_box(slide, "Fine-tuned delta1", f"{fine['delta1']:.4f}", 8.15, 1.2)
add_bullets(
    slide,
    [
        f"AbsRel changed by {comp['delta_abs_rel']:+.4f} (higher is worse here)",
        f"delta1 changed by {comp['delta_delta1']:+.4f} (lower is worse here)",
        "Conclusion: the small fine-tune overfit the PoC subset and hurt held-out test performance",
    ],
    0.8,
    3.0,
    7.2,
    2.0,
)
slide.shapes.add_picture(str(PLOTS / "test_absrel_histogram.png"), Inches(8.0), Inches(2.8), width=Inches(4.5))

# Slide 6
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Qualitative Example: Before Fine-Tuning")
slide.shapes.add_picture(str(PREDS / "poc_zero_shot" / "000000.png"), Inches(0.45), Inches(1.1), width=Inches(12.4))
add_caption(slide, "Zero-shot prediction panel: RGB | ground truth | predicted depth | error", 0.5, 6.7, 12.2)

# Slide 7
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Qualitative Example: After Fine-Tuning")
slide.shapes.add_picture(str(PREDS / "poc_finetuned" / "000000.png"), Inches(0.45), Inches(1.1), width=Inches(12.4))
add_caption(slide, "Fine-tuned prediction panel for the same sample", 0.5, 6.7, 12.2)

# Slide 8
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "System Assessment")
add_bullets(
    slide,
    [
        "Yes, this is a real and functioning depth-estimation system",
        "The pretrained zero-shot model produces plausible relative depth maps",
        "The system is accurate enough to demonstrate scene layout and nearer/farther structure",
        "The small supervised fine-tune did not improve it on held-out test data",
        "So the system works, but the training recipe needs improvement before claiming a better adapted model",
    ],
    0.8,
    1.4,
    11.7,
    3.6,
)
add_bullets(
    slide,
    [
        "Best claim: accurate relative depth proof of concept",
        "Not a strong claim: precise metric distance measurement",
    ],
    1.0,
    5.4,
    8.0,
    1.2,
    font_size=22,
)

# Slide 9
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Interactive Demo and Second Dataset")
add_bullets(
    slide,
    [
        "A local uploadable web app was added",
        "Run: ./run_upload_demo.sh",
        "Open: http://127.0.0.1:8000",
        "It runs the pretrained model on any uploaded image and saves the predicted depth map",
    ],
    0.8,
    1.3,
    6.0,
    2.5,
)
add_bullets(
    slide,
    [
        "Demo outputs are stored under outputs/web_demo/",
        "This is the most reliable demo mode because it uses the better zero-shot model",
    ],
    0.9,
    0.9,
    6.0,
    1.3,
    font_size=18,
)
add_bullets(
    slide,
    [
        "Second dataset:",
        "KITTI Tiny on Desktop",
        "7 outdoor driving frames",
        "Zero-shot folder inference worked successfully",
    ],
    7.3,
    1.4,
    4.3,
    2.0,
    font_size=18,
)
slide.shapes.add_picture(
    str(Path("/home/alexander/Desktop/DepthProjectDemo/KITTI_tiny/KITTI_tiny/2011_09_26/2011_09_26_drive_0023_sync/image_02/data/0000000000.png")),
    Inches(0.9),
    Inches(4.0),
    width=Inches(5.2),
)
slide.shapes.add_picture(
    str(Path("/home/alexander/Desktop/DepthProjectDemo/KITTI_tiny_depth_outputs/image_02/0000000000.png")),
    Inches(7.1),
    Inches(4.0),
    width=Inches(5.2),
)
add_caption(slide, "KITTI RGB sample", 0.9, 6.7, 5.2)
add_caption(slide, "KITTI predicted depth", 7.1, 6.7, 5.2)

# Slide 10
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_bg(slide)
add_title(slide, "Final Takeaway")
add_bullets(
    slide,
    [
        "This project meets the proof-of-concept goal for monocular relative depth estimation",
        "The pretrained model is the strongest result in this run",
        "The fine-tuned PoC provided useful evidence about overfitting on small subsets",
        "For a stronger adapted model, the next step is more data or a safer fine-tuning setup",
    ],
    0.8,
    1.7,
    11.6,
    3.2,
)


out_path = DESKTOP / "DepthProjectPresentation.pptx"
out_path.parent.mkdir(parents=True, exist_ok=True)
prs.save(out_path)
print(out_path)
