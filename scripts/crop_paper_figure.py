"""Crop all image blocks of a paper figure page and stack them into one PNG.

The KS qualitative-comparison figure is published as three separate raster blocks on
the page (one per method row), not one bitmap. Cropping only the largest block gets
just the reference row, as the first attempt showed. This script crops every image
block whose width is close to the widest one, orders them top to bottom, and pastes
them into a single image so the figure reads the way it does in the paper.

Usage:
    python scripts/crop_paper_figure.py --page 14 --stack
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PAPER = next((ROOT / "original papers").glob("Shen*.pdf"))
OUT = ROOT / "results" / "figures" / "paper_reference"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=int, default=14)
    ap.add_argument("--dpi", type=int, default=260)
    ap.add_argument("--pad", type=float, default=4.0)
    ap.add_argument("--stack", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PAPER)
    page = doc[a.page - 1]

    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 1]
    blocks.sort(key=lambda b: b["bbox"][1])  # top to bottom
    widest = max(b["bbox"][2] - b["bbox"][0] for b in blocks)
    kept = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) > 0.6 * widest]
    print(f"page {a.page}: {len(blocks)} image block(s), {len(kept)} kept")

    pixmaps = []
    for number, block in enumerate(kept, start=1):
        x0, y0, x1, y1 = block["bbox"]
        rect = pymupdf.Rect(x0 - a.pad, y0 - a.pad, x1 + a.pad, y1 + a.pad)
        pixmap = page.get_pixmap(clip=rect, dpi=a.dpi)
        pixmaps.append(pixmap)
        print(f"  row {number}: bbox={tuple(round(v, 1) for v in block['bbox'])} "
              f"-> {pixmap.width}x{pixmap.height} px")

    if a.stack and len(pixmaps) > 1:
        width = max(p.width for p in pixmaps)
        height = sum(p.height for p in pixmaps)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        row = 0
        for pixmap in pixmaps:
            array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n)[:, :, :3]
            offset = (width - pixmap.width) // 2
            canvas[row:row + pixmap.height, offset:offset + pixmap.width] = array
            row += pixmap.height
        path = OUT / f"paper_fig_page{a.page:02d}_stacked.png"
        pymupdf.Pixmap(pymupdf.csRGB, width, height, canvas.tobytes(), False).save(path)
        print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB, "
              f"{width}x{height} px)")
    else:
        for number, pixmap in enumerate(pixmaps, start=1):
            path = OUT / f"paper_fig_page{a.page:02d}_row{number}.png"
            pixmap.save(path)
            print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")

    text = page.get_text("text")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if ln.lower().startswith("figure")), None)
    if start is not None:
        print("\n--- caption ---")
        print(" ".join(lines[start:start + 12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
