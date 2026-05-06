from __future__ import annotations

import argparse
from pathlib import Path


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Depth Before After Gallery</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: #f3ede3; color: #1d1915; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 2.4rem; margin-bottom: 8px; }}
    p {{ color: #5c544f; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    .card {{ background: #fffaf1; border: 1px solid rgba(0,0,0,0.08); border-radius: 18px; padding: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); }}
    .card h2 {{ font-size: 1rem; margin: 0 0 10px; }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid rgba(0,0,0,0.08); }}
    .pair {{ display: grid; gap: 10px; }}
    .label {{ font-size: 0.9rem; color: #7c3d27; }}
  </style>
</head>
<body>
  <main>
    <h1>Depth Before / After Fine-Tuning</h1>
    <p>Each sample shows the zero-shot pretrained prediction panel and the post-training prediction panel.</p>
    <div class="grid">
      {cards}
    </div>
  </main>
</body>
</html>
"""


CARD = """<section class="card">
  <h2>{sample_id}</h2>
  <div class="pair">
    <div><div class="label">Before fine-tuning</div><img src="{before_src}" alt="before {sample_id}"></div>
    <div><div class="label">After fine-tuning</div><img src="{after_src}" alt="after {sample_id}"></div>
  </div>
</section>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    before_files = sorted(before_dir.glob("*.png"))
    for before_file in before_files:
        after_file = after_dir / before_file.name
        if not after_file.exists():
            continue
        cards.append(
            CARD.format(
                sample_id=before_file.stem,
                before_src=before_file.relative_to(output_path.parent).as_posix(),
                after_src=after_file.relative_to(output_path.parent).as_posix(),
            )
        )

    output_path.write_text(HTML.format(cards="\n".join(cards)), encoding="utf-8")


if __name__ == "__main__":
    main()
