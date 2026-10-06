"""Check API documentation coverage and, optionally, rendered local links.

Run from the repository root after installing the library. Uses the standard
library; --html additionally checks generated HTML anchors and local links.
"""

from __future__ import annotations

import argparse
import inspect
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import rgram
from rgram.aggregation import ArrayAggregation, Quantile, WeightedAggregation

ROOT = Path(__file__).resolve().parents[1]


def public_symbols():
    """Enumerate supported exports, adapter types, and all public class methods."""
    objects = {f"rgram.{name}": getattr(rgram, name) for name in rgram.__all__}
    objects.update(
        {
            f"rgram.aggregation.{cls.__name__}": cls
            for cls in (ArrayAggregation, WeightedAggregation, Quantile)
        }
    )
    for name, obj in list(objects.items()):
        if not inspect.isclass(obj) or issubclass(obj, Warning):
            continue
        for method, value in inspect.getmembers(obj):
            if callable(value) and (not method.startswith("_") or method == "__call__"):
                objects[f"{name}.{method}"] = value
    return objects


def check_source():
    """Require documented parameters, exports, modules, and fitted state."""
    pages = list((ROOT / "docs").rglob("*.md"))
    pages = [p for p in pages if "_build" not in p.parts]
    source = "\n".join(p.read_text() for p in pages)
    internals = (ROOT / "docs/internals.md").read_text()
    errors = []
    for module in (ROOT / "src/rgram").glob("*.py"):
        name = f"rgram.{module.stem}"
        if name not in internals:
            errors.append(f"Module missing from implementation guide: {name}")
    for name, obj in public_symbols().items():
        text = inspect.getdoc(obj) or ""
        if not text:
            errors.append(f"Missing docstring: {name}")
        if (
            name.count(".") == 1
            or name.startswith("rgram.aggregation.")
            and name.count(".") == 2
        ):
            if not re.search(
                r"(?:\{auto(?:class|function|exception)\}|auto(?:class|function|exception)::)\s+"
                + re.escape(name)
                + r"\b",
                source,
            ):
                errors.append(f"Missing API directive: {name}")
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        for arg, param in signature.parameters.items():
            if arg in ("self", "cls") or param.kind in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                continue
            if not re.search(r"(?m)^\s*" + re.escape(arg) + r"\s*:", text):
                errors.append(f"Undocumented parameter: {name}({arg})")
    # Check attributes assigned by Rgram code against class docs/result schemas.
    import ast

    result_text = (ROOT / "docs/api/results.md").read_text()
    for cls in (rgram.Regressogram, rgram.KernelSmoother, rgram.CoverageSearchCV):
        tree = ast.parse(inspect.getsource(cls))
        attributes = {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name)
            and n.value.id == "self"
            and not n.attr.startswith("_")
            and n.attr.endswith("_")
        }
        text = (inspect.getdoc(cls) or "") + result_text
        for name in attributes:
            if name not in text:
                errors.append(f"Undocumented fitted attribute: {cls.__name__}.{name}")
    return errors


class Page(HTMLParser):
    """Collect link destinations and fragment IDs without external dependencies."""

    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "name" in attrs:
            self.ids.add(attrs["name"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


def check_html(directory):
    """Require public API anchors and all local pages, fragments, and assets."""
    directory = directory.resolve()
    files = {
        p.resolve(): Page(p)
        for p in directory.rglob("*.html")
        if "_static" not in p.relative_to(directory).parts
    }
    errors = []
    ids = set().union(*(page.ids for page in files.values())) if files else set()
    for name in public_symbols():
        if name not in ids:
            errors.append(f"Missing rendered API anchor: {name}")
    if not (directory / ".nojekyll").exists():
        errors.append("Missing .nojekyll for GitHub Pages")
    for path, page in files.items():
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            local = unquote(url.path)
            if local.startswith("/"):
                errors.append(
                    f"Root-relative URL would break repository Pages: {path.name}: {href}"
                )
                continue
            target = (path.parent / local).resolve() if local else path
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                errors.append(
                    f"Broken local link: {path.relative_to(directory)}: {href}"
                )
            elif (
                url.fragment
                and target in files
                and unquote(url.fragment) not in files[target].ids
            ):
                errors.append(
                    f"Missing fragment: {path.relative_to(directory)}: {href}"
                )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="Built Sphinx HTML directory")
    args = parser.parse_args()
    errors = check_source()
    if args.html:
        errors.extend(check_html(args.html))
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print(
        f"Documentation covers {len(public_symbols())} public objects/methods and all package modules."
    )
    if args.html:
        print(
            "Rendered API anchors, local links/assets, and GitHub Pages paths passed."
        )


if __name__ == "__main__":
    main()
