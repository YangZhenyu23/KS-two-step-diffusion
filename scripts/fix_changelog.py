"""Repair the CHANGELOG version-heading sequence.

Editing this file by replacing the text just before ``## [vX.Y.Z]`` repeatedly eats the
previous version's heading, because the heading itself is part of the anchor. This
script checks that every version mentioned in the body has a matching ``## [v...]``
heading and, for a named version, re-inserts the heading in front of its summary line.

Usage:
    python scripts/fix_changelog.py --check
    python scripts/fix_changelog.py --restore 1.8.0 --before "新增双参数分布对比图"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "CHANGELOG.md"


def versions_with_headings(text: str) -> list[str]:
    return re.findall(r"^## \[v([0-9.]+)\]", text, flags=re.MULTILINE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--restore")
    ap.add_argument("--before", help="summary phrase that starts the version's body")
    ap.add_argument("--date", default="2026-11-12")
    a = ap.parse_args()

    text = LOG.read_text(encoding="utf-8")
    found = versions_with_headings(text)
    print("headings found:", ", ".join(found))

    if a.check:
        expected = [f"{major}.{minor}.0" for major in range(0, 3) for minor in range(0, 10)]
        missing = [v for v in expected if v not in found and re.search(rf"\bv{re.escape(v)}\b", text)]
        if missing:
            print("WARNING missing headings for:", ", ".join(missing))
        return 0

    if not a.restore or not a.before:
        ap.error("--restore needs --before")

    if a.restore in found:
        print(f"v{a.restore} already has a heading; nothing to do")
        return 0

    anchor = text.find(a.before)
    if anchor < 0:
        raise SystemExit(f"anchor phrase not found: {a.before!r}")

    # Walk back to the start of the paragraph (the blank line before the summary).
    insert_at = text.rfind("\n\n", 0, anchor) + 1
    heading = f"## [v{a.restore}] — {a.date}\n\n"
    text = text[:insert_at] + heading + text[insert_at:]
    LOG.write_text(text, encoding="utf-8")
    print(f"inserted heading for v{a.restore} at offset {insert_at}")
    print("headings now:", ", ".join(versions_with_headings(text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
