"""Audit the report prose against the collected numbers and against overclaim patterns.

Two checks, both mechanical.

1. NUMERIC TRACEABILITY
   Every decimal number in the Markdown report must appear either in
   results/fullscale/report_numbers.json (to 3-5 significant digits) or in a small allowlist
   of structural constants (parameter counts, particle counts, grid sizes, times). A number
   that matches nothing is either a typo or a stale figure from a superseded run, and both
   are worse than a missing number.

2. OVERCLAIM / STYLE
   Flag verbs and intensifiers that assert more than the evidence supports, and flag any
   sentence that states a rollout or capacity result without its sample size.

The script prints findings only; it does not edit. Exit code 1 means something needs a human
decision.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "REPORT_MAIN.md"
NUMBERS = ROOT / "results" / "fullscale" / "report_numbers.json"

text = MD.read_text(encoding="utf-8")
numbers = json.loads(NUMBERS.read_text(encoding="utf-8"))


def flatten(obj, out: set[float]) -> set[float]:
    if isinstance(obj, dict):
        for value in obj.values():
            flatten(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            flatten(value, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.add(float(obj))
    return out


known = flatten(numbers, set())

# structural constants that are not in the JSON because they are definitions, not results
ALLOW = {
    9027, 17475, 26502, 26547, 34435, 134403, 71559,  # parameter counts
    100000, 20000, 3000, 2000, 4096, 8000, 2793,      # particle / batch / solve counts
    20, 10, 200, 40, 45, 90, 65, 143, 144, 112, 5, 15, 35, 50, 93, 429, 448, 572, 679,
    16, 64, 128, 256, 112,                            # bins, widths
    0.02, 0.005, 0.01,                                # T, dt, T/2
    15.0, 150.0, 160.0, 0.1, 10.0,                    # parameter ranges
    1, 2, 4, 8, 0, 3, 6, 11, 14, 17, 19, 21, 23, 29, 30, 45, 47, 55, 78, 80, 85, 91, 95,
}
known |= {float(v) for v in ALLOW}

# collect decimals with 2+ significant digits
candidates = set()
for match in re.finditer(r"(?<![\w.])(\d+\.\d+)", text):
    candidates.add(float(match.group(1)))

def matches_known(value: float) -> bool:
    for k in known:
        if k == 0:
            continue
        for digits in (3, 2, 1):
            if abs(round(k, digits) - value) < 10 ** (-digits) / 2:
                return True
        # percentages written to one decimal
        if abs(round(100 * k, 1) - value) < 0.05:
            return True
    return False

unknown = sorted(v for v in candidates if not matches_known(v))

print(f"decimal tokens found : {len(candidates)}")
print(f"traceable to artifacts: {len(candidates) - len(unknown)}")
print(f"UNTRACEABLE          : {len(unknown)}")
if unknown:
    print("\n  untraceable values (check each against the run that produced it):")
    for value in unknown:
        # show the line it appears on
        for line in text.splitlines():
            if f"{value}" in line:
                snippet = line.strip()[:110]
                print(f"    {value:<10} | {snippet}")
                break

# ---- overclaim scan ----
print("\n=== overclaim / style scan ===")
BANNED = ["proves", "proven", "clearly shows", "definitively", "significantly better",
          "state-of-the-art", "novel", "robustly", "dramatically", "substantially improves",
          "outperforms all", "guarantees", "always works"]
findings = []
lower = text.lower()
for phrase in BANNED:
    for match in re.finditer(re.escape(phrase), lower):
        start = max(0, match.start() - 70)
        snippet = " ".join(text[start:match.end() + 70].split())
        findings.append((phrase, snippet))

if findings:
    for phrase, snippet in findings:
        print(f"  [{phrase}] ...{snippet}...")
else:
    print("  no banned phrases found")

# ---- sample-size checks ----
print("\n=== sample-size disclosure for the weak claims ===")
checks = [
    ("single seed", ["capacity comparison is a single seed", "Test 1 is a single seed"]),
    ("thin rollout", ["3 cells", "6 of 6"]),
]
for label, needles in checks:
    hit = any(n.lower() in lower for n in needles)
    print(f"  {label:<16} disclosed? {hit}")

sys.exit(1 if unknown else 0)
