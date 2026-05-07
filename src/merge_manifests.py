from __future__ import annotations

import argparse
from pathlib import Path


def load_rows(path: Path) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        rows.append(line)
        seen.add(line)
    return rows


def absolutize_row(manifest_path: Path, row: str) -> str:
    parts = row.split()
    if len(parts) < 2:
        raise ValueError(f"Manifest row must contain image and depth paths: {row}")
    root = manifest_path.parent.parent if manifest_path.parent.name == "splits" else manifest_path.parent
    resolved: list[str] = []
    for part in parts[:2]:
        path = Path(part)
        if not path.is_absolute():
            path = (root / path).resolve()
        resolved.append(path.as_posix())
    if len(parts) > 2:
        resolved.extend(parts[2:])
    return " ".join(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple depth manifests into one absolute-path manifest.")
    parser.add_argument("--output", required=True, help="Output manifest path")
    parser.add_argument(
        "--manifest",
        dest="manifests",
        action="append",
        required=True,
        help="Manifest to merge. Pass multiple times in the order you want datasets added.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Rewrite relative rows to absolute image/depth paths so merged manifests can span multiple roots.",
    )
    args = parser.parse_args()

    merged_rows: list[str] = []
    seen: set[str] = set()
    for manifest_arg in args.manifests:
        manifest_path = Path(manifest_arg).resolve()
        for row in load_rows(manifest_path):
            out_row = absolutize_row(manifest_path, row) if args.absolute_paths else row
            if out_row in seen:
                continue
            merged_rows.append(out_row)
            seen.add(out_row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(merged_rows) + ("\n" if merged_rows else ""), encoding="utf-8")
    print(f"wrote {len(merged_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
