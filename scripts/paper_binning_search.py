"""Search the published paper for how the projection heatmaps are specified.

The figure code in the upstream repo computes probability *density*
(``counts / N / dx / dy``), but the colour bars rendered in the paper reach ~25-30
on 100k particles, which looks like counts, not density. This matters for reproducing
the figure faithfully: a density colour bar and a count colour bar differ by a factor
of ``N * dx * dy``, and the paper's own figure has to be read with the right units.

This script greps every caption and paragraph for the relevant vocabulary and prints
the surrounding text so the ambiguity can be settled from the paper rather than assumed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PAPER = next((ROOT / "original papers").glob("Shen*.pdf"))

KEYWORDS = ("histogram", "bins", "bin ", "colorbar", "color bar", "colour bar",
            "density map", "jet", "viridis", "heat map", "heatmap", "kernel density")


def main() -> int:
    doc = pymupdf.open(PAPER)
    hits = 0
    for number in range(len(doc)):
        text = doc[number].get_text()
        for keyword in KEYWORDS:
            for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                start = max(0, match.start() - 220)
                end = min(len(text), match.end() + 220)
                snippet = " ".join(text[start:end].split())
                print(f"p{number + 1}  [{keyword.strip()}]  ...{snippet}...\n")
                hits += 1
    print(f"total matches: {hits}")
    if hits == 0:
        print("\nThe paper does not specify the histogram/colour-bar convention.")
        print("=> The units of the published colour bars cannot be settled from the text;")
        print("   the only authoritative source is the upstream plotting code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
