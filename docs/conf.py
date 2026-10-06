"""Sphinx configuration for the GitHub Pages documentation."""

from importlib.metadata import version as package_version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "Rgram"
author = "Rgram contributors"
copyright = "2026, Rgram contributors"
release = package_version("rgram")
version = release
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "requirements.txt"]
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 4
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
# Attribute types are readable text, rather than links to every typing alias.
napoleon_use_ivar = False
html_theme = "pydata_sphinx_theme"
html_title = "Rgram documentation"
html_static_path = ["_static"]
html_css_files = ["rgram.css"]
html_theme_options = {
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
}
html_show_sourcelink = True

viewcode_follow_imported_members = False
