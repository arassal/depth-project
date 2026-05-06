from __future__ import annotations

import io
import uuid
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


APP_ROOT = Path("/home/alexander/depth-project")
ARTIFACT_ROOT = APP_ROOT / "outputs" / "web_demo"
UPLOAD_DIR = ARTIFACT_ROOT / "uploads"
DEPTH_DIR = ARTIFACT_ROOT / "depth"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "LiheYoung/depth-anything-small-hf"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)
MODEL = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE).eval()

app = Flask(__name__)


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Depth Upload Demo</title>
  <style>
    :root {
      --bg: #f4efe5;
      --card: #fffaf1;
      --ink: #1f1a17;
      --muted: #645a54;
      --accent: #a84425;
      --line: rgba(0,0,0,0.1);
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: linear-gradient(180deg,#f8f2e8,#eee1cf); color: var(--ink); font-family: Georgia, serif; }
    main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 60px; }
    h1 { font-size: 3rem; line-height: 0.95; margin: 0 0 12px; }
    p { color: var(--muted); line-height: 1.6; }
    .hero, .panel, .card { background: rgba(255,250,241,0.92); border: 1px solid var(--line); border-radius: 24px; box-shadow: 0 16px 40px rgba(0,0,0,0.08); }
    .hero { padding: 28px; margin-bottom: 22px; }
    .grid { display: grid; gap: 20px; grid-template-columns: 380px 1fr; }
    .panel { padding: 22px; }
    form { display: grid; gap: 14px; }
    input[type=file] { width: 100%; padding: 16px; background: #fff; border: 1px dashed var(--line); border-radius: 18px; }
    button { padding: 14px 18px; background: var(--accent); color: white; border: 0; border-radius: 14px; font-size: 1rem; cursor: pointer; }
    .meta { display: grid; gap: 8px; margin-top: 16px; }
    .pill { display: inline-block; padding: 7px 10px; background: rgba(168,68,37,0.1); border: 1px solid rgba(168,68,37,0.14); border-radius: 999px; color: var(--accent); font-size: 0.9rem; }
    .cards { display: grid; gap: 18px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card { padding: 16px; }
    .card h2 { margin: 0 0 10px; font-size: 1.05rem; }
    img { width: 100%; border-radius: 16px; border: 1px solid var(--line); display: block; background: #fff; }
    .recent { margin-top: 24px; }
    .recent-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .recent-grid a { text-decoration: none; color: inherit; }
    .thumb { background: rgba(255,250,241,0.92); border: 1px solid var(--line); border-radius: 18px; padding: 10px; }
    .thumb span { display: block; margin-top: 8px; font-size: 0.9rem; color: var(--muted); }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .cards { grid-template-columns: 1fr; }
      h1 { font-size: 2.3rem; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Upload an image and run monocular depth estimation.</h1>
      <p>This uses the pretrained Depth Anything Small checkpoint directly. The uploaded image stays local, inference runs on this machine, and the page saves both the original and predicted depth map into the project outputs.</p>
    </section>

    <div class="grid">
      <section class="panel">
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="image" accept="image/png,image/jpeg,image/webp" required>
          <button type="submit">Run Depth Model</button>
        </form>
        <div class="meta">
          <span class="pill">Model: Depth Anything Small</span>
          <span class="pill">Mode: zero-shot inference</span>
          <span class="pill">Device: {{ device }}</span>
        </div>
        {% if result %}
        <p><strong>Saved sample:</strong> {{ result.sample_id }}</p>
        {% endif %}
      </section>

      <section class="panel">
        {% if result %}
        <div class="cards">
          <article class="card">
            <h2>Original image</h2>
            <img src="{{ url_for('artifact', kind='uploads', filename=result.original_name) }}" alt="original upload">
          </article>
          <article class="card">
            <h2>Predicted depth map</h2>
            <img src="{{ url_for('artifact', kind='depth', filename=result.depth_name) }}" alt="predicted depth">
          </article>
        </div>
        {% else %}
        <p>No upload yet. Choose an image and run the model.</p>
        {% endif %}
      </section>
    </div>

    {% if recent %}
    <section class="recent">
      <h2>Recent runs</h2>
      <div class="recent-grid">
        {% for item in recent %}
        <a href="{{ url_for('index', sample=item.sample_id) }}">
          <div class="thumb">
            <img src="{{ url_for('artifact', kind='depth', filename=item.depth_name) }}" alt="{{ item.sample_id }}">
            <span>{{ item.sample_id }}</span>
          </div>
        </a>
        {% endfor %}
      </div>
    </section>
    {% endif %}
  </main>
</body>
</html>
"""


def recent_items(limit: int = 8) -> list[dict[str, str]]:
    items = []
    for depth_file in sorted(DEPTH_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        sample_id = depth_file.stem
        original = UPLOAD_DIR / f"{sample_id}.png"
        if original.exists():
            items.append({"sample_id": sample_id, "original_name": original.name, "depth_name": depth_file.name})
    return items


def run_depth(image: Image.Image) -> np.ndarray:
    inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        pred = MODEL(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    pred = pred.detach().cpu().numpy()
    pred = pred - pred.min()
    pred = pred / (pred.max() + 1e-8)
    return pred


def colorize_depth(depth: np.ndarray) -> Image.Image:
    colored = cm.get_cmap("plasma")(depth)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored)


def save_upload_and_depth(file_storage) -> dict[str, str]:
    image = Image.open(file_storage.stream).convert("RGB")
    sample_id = uuid.uuid4().hex[:12]
    original_name = f"{sample_id}.png"
    depth_name = f"{sample_id}.png"

    original_path = UPLOAD_DIR / original_name
    depth_path = DEPTH_DIR / depth_name

    image.save(original_path)
    depth = run_depth(image)
    colorize_depth(depth).save(depth_path)

    return {"sample_id": sample_id, "original_name": original_name, "depth_name": depth_name}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            result = save_upload_and_depth(image_file)
            return redirect(url_for("index", sample=result["sample_id"]))

    sample_id = request.args.get("sample")
    if sample_id:
        original = UPLOAD_DIR / f"{sample_id}.png"
        depth = DEPTH_DIR / f"{sample_id}.png"
        if original.exists() and depth.exists():
            result = {"sample_id": sample_id, "original_name": original.name, "depth_name": depth.name}

    return render_template_string(PAGE, result=result, recent=recent_items(), device=str(DEVICE))


@app.route("/artifacts/<kind>/<path:filename>")
def artifact(kind: str, filename: str):
    if kind == "uploads":
      return send_from_directory(UPLOAD_DIR, filename)
    if kind == "depth":
      return send_from_directory(DEPTH_DIR, filename)
    return ("not found", 404)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
