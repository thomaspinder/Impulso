"""Post-process Quarto-rendered markdown for Zensical compatibility.

Fixes two issues with Quarto GFM output:

1. DataFrame tables: Quarto wraps pandas DataFrame pipe tables in
   <div><style scoped>...</style>...</div> blocks.  Python-Markdown does not
   parse markdown inside raw HTML blocks, so the pipe tables render as literal
   text.  This script strips the wrapper, leaving bare pipe tables.

2. Display math: With ``wrap: none``, Quarto puts ``$$...$$`` inline on long
   lines.  pymdownx.arithmatex misparses inline ``$$`` as two consecutive ``$``
   delimiters.  This script moves ``$$...$$`` onto dedicated lines so arithmatex
   recognises them as display math.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Pattern: <div>\n<style scoped>\n  ...dataframe...\n</style>\n\n(table)\n\n</div>
_DATAFRAME_DIV = re.compile(
    r"<div>\s*\n<style scoped>\s*\n.*?\.dataframe.*?</style>\s*\n\n(.*?)\n\n</div>",
    re.DOTALL,
)

# Pattern: inline $$...$$  (not already on its own line)
_INLINE_DISPLAY_MATH = re.compile(
    r"(?<!\n)\$\$(.+?)\$\$(?!\n)",
    re.DOTALL,
)


def postprocess(text: str) -> str:
    # Strip <div>/<style scoped> wrappers around DataFrame pipe tables
    text = _DATAFRAME_DIV.sub(r"\1", text)

    # Move inline $$...$$ onto their own lines
    text = _INLINE_DISPLAY_MATH.sub(r"\n\n$$\1$$\n\n", text)

    return text


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: postprocess_qmd.py <file_or_glob> ...")
        sys.exit(1)

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(Path(".").glob(arg))

    if not paths:
        print("No matching files found.")
        return

    for path in paths:
        original = path.read_text()
        result = postprocess(original)
        if result != original:
            path.write_text(result)
            print(f"  patched: {path}")
        else:
            print(f"  no changes: {path}")


if __name__ == "__main__":
    main()
