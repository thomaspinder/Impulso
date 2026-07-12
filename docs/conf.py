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
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_codeautolink",  # link API names in example code to the reference
    "sphinx_sitemap",  # emit sitemap.xml
]

# -- Bibliography (sphinxcontrib-bibtex) -------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"  # {cite:t} -> Uhlig (2005)
bibtex_default_style = "unsrt"

templates_path = ["_templates"]

# Tutorials are jupytext py:percent notebooks; MyST-NB reads them via jupytext.
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".py": "myst-nb",
    ".rst": "restructuredtext",  # autosummary-generated API stubs
}
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "conf.py",  # this config module is not a document
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
# Smoke (IMPULSO_DOCS_CI=1) and full renders execute the SAME notebook source
# but with very different MCMC — yet jupyter-cache keys on source alone, so a
# shared cache lets a tiny smoke run (draws=10) poison the full-fidelity build.
# Separate the cache directories so the two modes can never overwrite one
# another (locally or in CI).
_smoke_render = os.environ.get("IMPULSO_DOCS_CI") == "1"
nb_execution_cache_path = os.path.join(
    os.path.dirname(__file__),
    "_build",
    ".jupyter_cache_ci" if _smoke_render else ".jupyter_cache",
)
nb_execution_timeout = 1800  # heavy MCMC tutorials
# Strict by default (PR gate + local): a failed cell fails the build. On the
# deploy path (IMPULSO_DOCS_RESILIENT=1) we do NOT raise, so one slow/broken
# MCMC notebook cannot block the whole site — it renders an error cell while
# every other page (and the last-good cached output) still deploys.
nb_execution_raise_on_error = os.environ.get("IMPULSO_DOCS_RESILIENT") != "1"
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

# -- Numbered figures, tables and equations ----------------------------------
numfig = True
math_eqref_format = "Eq. {number}"

# -- sphinx-codeautolink -----------------------------------------------------
# Adds a "Examples using …" backreference block to each documented object.
codeautolink_autodoc_inject = True

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
html_theme = "shibuya"
html_title = "impulso"
html_baseurl = "https://thomaspinder.github.io/impulso/"  # for sitemap + canonical
sitemap_url_scheme = "{link}"
html_static_path = ["stylesheets"]
html_css_files = ["extra.css"]
html_theme_options = {
    "accent_color": "crimson",  # radix name closest to the brand #870d14
    "color_mode": "auto",  # follow the reader's light/dark preference
    "github_url": "https://github.com/thomaspinder/impulso",
    "nav_links": [
        {"title": "PyPI", "url": "https://pypi.org/project/impulso"},
    ],
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
