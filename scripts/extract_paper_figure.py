"""Locate and render the projection figure from the original Shen et al. paper.

Why: ``scripts/projection_figure.py`` was written by reading the upstream *code*
(``ks.py::_projections``) and the upstream *reimplementation* PDF. That is one step
removed from the paper itself. This script finds the page that actually carries the
3D-distribution figure in the published PDF and renders it, so the imitation can be
checked against the original rather than against a re-drawing of it.

The figure is located by text: the caption of the KS projection figure is the one
that mentions both a reference and a refined/generated distribution together with
plane/projection language. Page-level keyword scoring is enough here because the
paper is short and the figure caption sits on the same page as the figure.

Usage:
    python scripts/extract_paper_figure.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PAPER = next((ROOT / "original papers").glob("Shen*.pdf"))
OUT = ROOT / "results" / "figures" / "paper_reference"

KEYWORDS = ("reference", "projection", "profiles", "density", "x-y", "x-z", "y-z",
            "refinement", "generated")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PAPER)
    print(f"paper: {PAPER.name}  pages={len(doc)}")

    scored = []
    for number, page in enumerate(doc):
        text = page.get_text().lower()
        hits = sum(1 for word in KEYWORDS if word in text)
        images = len(page.get_images(full=True))
        drawings = len(page.get_drawings())
        scored.append((hits, images, drawings, number))
    scored.sort(reverse=True)

    print(f"\n{'page':>5} {'hits':>5} {'images':>7} {'drawings':>9}")
    for hits, images, drawings, number in scored[:8]:
        print(f"{number + 1:5d} {hits:5d} {images:7d} {drawings:9d}")

    # Render the best few candidates at a readable resolution.
    written = []
    for rank, (hits, images, drawings, number) in enumerate(scored[:4], start=1):
        page = doc[number]
        pixmap = page.get_pixmap(dpi=170)
        path = OUT / f"paper_page{number + 1:02d}_rank{rank}.png"
        pixmap.save(path)
        written.append(path)
        print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")

        # Also dump the caption text so the figure can be identified without guessing.
        text = page.get_text()
        captions = re.findall(r"Fig(?:ure)?\.?\s*\d+[^\n]{0,400}", text, flags=re.IGNORECASE)
        for caption in captions[:3]:
            print("    caption:", " ".join(caption.split())[:220])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
