"""Generate standalone gallery sources and page code from the maintained examples."""

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = (ROOT / "examples/plot_relationships.py").read_text()
    tree = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    header = source[source.index("from __future__") : source.index("def bins()")]
    for page in sorted((ROOT / "docs/gallery").glob("*.md")):
        if page.stem not in functions:
            continue
        code = (
            header
            + functions[page.stem]
            + '\n\nif __name__ == "__main__":\n    import matplotlib.pyplot as plt\n\n    figure = '
            + page.stem
            + "()\n    plt.show()\n    plt.close(figure)\n"
        )
        destination = ROOT / "docs/_static/examples" / f"{page.stem}.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = page.read_text()
        marker = (
            "Download [the complete gallery script]"
            if "Download [the complete gallery script]" in text
            else "Download [this standalone example]"
        )
        text = text[: text.index(marker)] + (
            f"Download [this standalone example](../_static/examples/{page.stem}.py).\n"
            "The complete code below includes imports, data, configuration, and plotting.\n"
            f"Run the downloaded file with `python {page.stem}.py`.\n\n"
            "```python\n" + code + "```\n"
        )
        for path, content in ((destination, code), (page, text)):
            if args.check:
                if not path.exists() or path.read_text() != content:
                    raise SystemExit(f"Stale gallery artifact: {path}")
            else:
                path.write_text(content)
    prediction = ROOT / "docs/gallery/prediction.md"
    code = (ROOT / "examples/held_out_prediction.py").read_text()
    text = prediction.read_text()
    updated = re.sub(
        r"```python\n.*?```", lambda _: "```python\n" + code + "```", text, flags=re.S
    )
    download = ROOT / "docs/_static/examples/held_out_prediction.py"
    for path, content in ((prediction, updated), (download, code)):
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Stale prediction artifact: {path}")
        else:
            path.write_text(content)
    print("Verified ten standalone gallery sources and displayed code.")


if __name__ == "__main__":
    main()
