"""Emit the GitHub Pages marker after each MkDocs build."""

from pathlib import Path


def on_post_build(config, **kwargs):
    (Path(config["site_dir"]) / ".nojekyll").touch()
