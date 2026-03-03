"""MkDocs hooks for impulso documentation."""

import subprocess
import sys
from pathlib import Path

TUTORIALS_DIR = Path(__file__).parent / "tutorials"
NOTEBOOKS = [
    "forecasting.py",
    "structural-analysis.py",
]


def on_pre_build(config):
    """Export marimo tutorial notebooks to static markdown before building docs."""
    for notebook in NOTEBOOKS:
        src = TUTORIALS_DIR / notebook
        dst = TUTORIALS_DIR / src.with_suffix(".md").name
        subprocess.run(
            [sys.executable, "-m", "marimo", "export", "md", str(src), "-o", str(dst), "-f"],
            check=True,
        )
