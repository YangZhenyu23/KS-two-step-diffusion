"""Prepare the pandoc HTML for print: render the Mermaid block in the browser."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs"
SCRIPT = """
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
mermaid.initialize({
  startOnLoad: true,
  theme: "neutral",
  securityLevel: "loose",
  fontFamily: "Segoe UI, Microsoft YaHei, sans-serif"
});
</script>
"""
PATTERN = re.compile(r'<pre class="mermaid"><code>(.*?)</code></pre>', re.S)


def patch(name: str) -> None:
    path = ROOT / name
    text = path.read_text(encoding="utf-8")
    text, count = PATTERN.subn(
        lambda match: '<div class="mermaid">' + match.group(1) + "</div>",
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"{name}: expected 1 mermaid block, found {count}")
    if "</body>" not in text:
        raise SystemExit(f"{name}: no body close tag")
    text = text.replace("</body>", SCRIPT + "\n</body>", 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {name}")


if __name__ == "__main__":
    patch("HARRY_NOTE.preview.html")
    patch("HARRY_NOTE.zh.preview.html")
