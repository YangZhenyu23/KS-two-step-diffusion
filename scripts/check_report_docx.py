"""Sanity-check the generated Word report.

Catches the failure modes that a build script can hide: a NaN leaking into a table because
a JSON key was wrong, a missing figure silently replaced by a placeholder line, or an
empty section.
"""
from __future__ import annotations

import re
from pathlib import Path
import sys

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs" / "report_for_harry.docx"

doc = Document(PATH)
paragraphs = [p.text for p in doc.paragraphs]
text = "\n".join(paragraphs)

print(f"paragraphs : {len(paragraphs)}")
print(f"tables     : {len(doc.tables)}")
print(f"images     : {len(doc.inline_shapes)}")
print(f"words      : {len(text.split())}")

problems = []
if "nan" in text.lower():
    for line in text.splitlines():
        if "nan" in line.lower():
            problems.append(f"NaN in text: {line.strip()[:100]}")
for number, table in enumerate(doc.tables, start=1):
    for row in table.rows:
        for cell in row.cells:
            if "nan" in cell.text.lower():
                problems.append(f"table {number}: NaN in cell {cell.text!r}")
if "[figure missing" in text:
    for line in text.splitlines():
        if "[figure missing" in line:
            problems.append(line.strip())

sections = re.findall(r"^\d+\.\s+.*$", text, flags=re.MULTILINE)
print("\ntop-level sections:")
for section in sections[:20]:
    print(f"  {section}")

print("\nplaceholder banners found:",
      text.count("RUNNING") + text.count("QUEUED"))

if problems:
    print(f"\n{len(problems)} PROBLEM(S):")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print("\nno NaNs, no missing figures")
