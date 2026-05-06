from __future__ import annotations

import argparse
from pathlib import Path


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: #f4eee3; color: #1e1915; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 2.4rem; margin-bottom: 6px; }}
    p {{ color: #645a54; }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
    .card {{ background: #fffaf1; border: 1px solid rgba(0,0,0,0.08); border-radius: 18px; padding: 12px; box-shadow: 0 12px 32px rgba(0,0,0,0.06); }}
    .pair {{ display: grid; gap: 10px; }}
    img {{ width: 100%; display: block; border-radius: 12px; border: 1px solid rgba(0,0,0,0.08); }}
    .label {{ font-size: 0.92rem; color: #8a3f26; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p>{subtitle}</p>
    <div class="grid">{cards}</div>
  </main>
</body>
</html>
"""


CARD = """<section class="card">
  <h2>{sample_id}</h2>
  <div class="pair">
    <div><div class="label">{left_label}</div><img src="{left_src}" alt="{sample_id} left"></div>
    <div><div class="label">{right_label}</div><img src="{right_src}" alt="{sample_id} right"></div>
  </div>
</section>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Side by Side Gallery")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--left-label", default="Original")
    parser.add_argument("--right-label", default="Prediction")
    args = parser.parse_args()

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    for left_file in sorted(left_dir.glob("*.png")):
        right_file = right_dir / left_file.name
        if not right_file.exists():
            continue
        cards.append(
            CARD.format(
                sample_id=left_file.stem,
                left_src=left_file.resolve().as_uri(),
                right_src=right_file.resolve().as_uri(),
                left_label=args.left_label,
                right_label=args.right_label,
            )
        )

    output.write_text(
        HTML.format(
            title=args.title,
            subtitle=args.subtitle,
            cards="\n".join(cards),
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
