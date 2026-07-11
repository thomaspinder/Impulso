"""One-shot migration: Quarto .qmd tutorial -> MyST-NB executable .md.

Pipeline per file:  quarto convert (qmd -> ipynb)  ->  jupytext (ipynb -> md:myst)
This script post-processes the jupytext output to finish the job:

1. Strip the embedded Quarto YAML block that `quarto convert` leaves in the body,
   lifting its ``title`` into a top-level ``# H1``.
2. Translate Quarto per-cell ``#| key: value`` directives into MyST-NB cell
   ``:tags:`` (``include: false`` -> remove-cell, ``echo: false`` -> remove-input,
   ``output: false`` -> remove-output). Unmapped directives are dropped.

This is a build-time authoring transform run ONCE during migration, not a
render-time patch on generated HTML. After migration the .md is the source.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

_DIRECTIVE_TO_TAG = {
    ("include", "false"): "remove-cell",
    ("echo", "false"): "remove-input",
    ("output", "false"): "remove-output",
}


def _strip_quarto_frontmatter(body: str) -> tuple[str, str | None]:
    """Remove the embedded Quarto ``---...---`` block; return (body, title)."""
    m = re.match(r"\s*---\n(.*?)\n---\n", body, re.DOTALL)
    if not m:
        return body, None
    block = m.group(1)
    title = None
    tm = re.search(r"^title:\s*(.+?)\s*$", block, re.MULTILINE)
    if tm:
        title = tm.group(1).strip().strip("\"'")
    # only strip if it actually looks like the Quarto header (has format/jupyter/title)
    if re.search(r"^(format|jupyter|execute|title):", block, re.MULTILINE):
        body = body[m.end() :]
    return body, title


def _convert_cell(header: str, code: str) -> str:
    """Map leading ``#|`` directives in a code cell to MyST ``:tags:``."""
    lines = code.splitlines()
    tags: list[str] = []
    keep: list[str] = []
    consuming = True
    for line in lines:
        dm = re.match(r"\s*#\|\s*([\w-]+):\s*(.+?)\s*$", line) if consuming else None
        if dm:
            key, val = dm.group(1), dm.group(2).strip()
            if key == "tags":
                tags.extend(re.findall(r"[\w-]+", val))
            elif (key, val) in _DIRECTIVE_TO_TAG:
                tags.append(_DIRECTIVE_TO_TAG[(key, val)])
            # else: drop (warning/message/label/fig-cap/output: asis handled manually)
            continue
        if line.strip():
            consuming = False
        keep.append(line)
    body = "\n".join(keep).strip("\n")
    tagline = f":tags: [{', '.join(dict.fromkeys(tags))}]\n" if tags else ""
    return f"{header}\n{tagline}{body}\n```"


def postprocess(text: str) -> str:
    # split frontmatter (jupytext) from body
    fm = re.match(r"(---\n.*?\n---\n)", text, re.DOTALL)
    front = fm.group(1) if fm else ""
    body = text[len(front) :]
    body, title = _strip_quarto_frontmatter(body)

    # rewrite each code-cell block
    def repl(m: re.Match) -> str:
        return _convert_cell(m.group(1), m.group(2))

    body = re.sub(r"(```\{code-cell\}[^\n]*)\n(.*?)\n```", repl, body, flags=re.DOTALL)

    heading = f"\n# {title}\n" if title else ""
    return f"{front}{heading}{body.lstrip(chr(10))}"


def migrate(qmd: Path) -> Path:
    subprocess.run(["quarto", "convert", str(qmd)], check=True, capture_output=True)
    ipynb = qmd.with_suffix(".ipynb")
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as tmp:
        subprocess.run(
            ["jupytext", "--to", "md:myst", str(ipynb), "-o", tmp.name],
            check=True,
            capture_output=True,
        )
        raw = Path(tmp.name).read_text()
    out = qmd.with_suffix(".md")
    out.write_text(postprocess(raw))
    ipynb.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = migrate(Path(arg))
        print(f"  migrated: {p}")
