"""Sphinx configuration for the Impulso documentation.

Single-engine build: MyST-NB parses and executes the tutorials, Sphinx renders
everything. No Quarto, no pandoc, no markdown post-processing.
"""

from __future__ import annotations

import os

# -- Project information -----------------------------------------------------
project = "impulso"
author = "Thomas Pinder"
copyright = "2026, Thomas Pinder"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",  # MyST markdown + executable notebooks (enables myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "agents/**",  # internal agent docs, not part of the site
    "adr/**",  # architecture decision records live in-repo, not on the site
]

# -- MyST / MyST-NB ----------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",  # $...$ and $$...$$ — the feature Quarto+Zensical broke
    "amsmath",  # \begin{align} etc.
    "colon_fence",  # ::: {note} admonitions
    "deflist",
    "tasklist",
    "html_image",
    "attrs_inline",
    "substitution",
]
myst_dmath_double_inline = True
myst_heading_anchors = 3  # auto-slug headings so in-page [](#anchor) links resolve

# Execute notebooks and cache results in a gitignored cache. Source .md carry
# no outputs; a cell re-runs only when its code changes.
nb_execution_mode = "cache"
nb_execution_timeout = 1800  # heavy MCMC tutorials
nb_execution_raise_on_error = True
nb_merge_streams = True

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "impulso"
html_static_path = ["stylesheets"]
html_css_files = ["extra.css"]
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/thomaspinder/impulso",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/impulso",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navigation_with_keys": True,
}
html_context = {
    "github_user": "thomaspinder",
    "github_repo": "impulso",
    "github_version": "main",
    "doc_path": "docs",
}

# Disable the sampler progress widget in all rendered notebooks (inherited by
# the myst-nb execution kernel).
os.environ["IMPULSO_DOCS_BUILD"] = "1"

# Smoke-render flag for CI (mirrors the old Quarto `ci` parameter). Tutorials
# read this to shrink MCMC when set.
os.environ.setdefault("IMPULSO_DOCS_CI", "0")
