"""Execute the documentation's Python examples and public API doctests headlessly."""

from __future__ import annotations

from contextlib import redirect_stdout
import doctest
import importlib
import importlib.util
from io import StringIO
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


def main():
    pages = sorted(
        p.relative_to(ROOT / "docs").as_posix()
        for p in (ROOT / "docs").rglob("*.md")
        if "_build" not in p.parts
    )
    blocks = rendered_attempts = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ["../README.md", *pages]:
            namespace = {"__name__": "__main__"}
            text = (ROOT / "docs" / name).read_text()
            for match in re.finditer(
                r"```python\n(.*?)```(?:\s*```text\n(.*?)```)?", text, re.S
            ):
                code, expected = match.groups()
                # Keep figures/tables out of CI logs; failures still propagate.
                captured = StringIO()
                with redirect_stdout(captured):
                    exec(compile(code, f"docs/{name}", "exec"), namespace)
                if (
                    expected is not None
                    and captured.getvalue().strip() != expected.strip()
                ):
                    raise AssertionError(
                        f"Displayed output differs: {name}: {captured.getvalue()}"
                    )
                blocks += 1
            for code in re.findall(r"```pycon\n(.*?)```", text, re.S):
                test = doctest.DocTestParser().get_doctest(code, {}, name, name, 0)
                runner = doctest.DocTestRunner()
                runner.run(test)
                rendered_attempts += runner.tries
                if runner.failures:
                    raise AssertionError(f"Rendered doctest failed: {name}")
            import matplotlib.pyplot as plt

            for number in plt.get_fignums():
                plt.figure(number).canvas.draw()
            plt.close("all")
        spec = importlib.util.spec_from_file_location(
            "rgram_gallery", ROOT / "examples/plot_relationships.py"
        )
        gallery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gallery)
        import matplotlib.pyplot as plt

        for create in gallery.EXAMPLES.values():
            figure = create()
            assert len(figure.axes) == 1, "Gallery plots must have one panel"
            assert tuple(figure.get_size_inches()) == (10, 6.6)
            # Draw every figure to catch plotting errors in headless CI.
            figure.canvas.draw()
            from io import BytesIO
            import numpy as np
            from matplotlib.image import imread

            rendered = BytesIO()
            figure.savefig(rendered, format="png", dpi=180)
            rendered.seek(0)
            actual = imread(rendered)
            expected = imread(ROOT / "docs/_static/gallery" / f"{create.__name__}.png")
            if not np.array_equal(actual, expected):
                raise AssertionError(f"Stale gallery image: {create.__name__}")
            plt.close(figure)
        failures = attempts = 0
        for module in (
            "rgram.base",
            "rgram.rgram",
            "rgram.smoothing",
            "rgram.model_selection",
            "rgram.aggregation",
            "rgram.plotting",
        ):
            result = doctest.testmod(importlib.import_module(module))
            failures += result.failed
            attempts += result.attempted
    if failures:
        raise SystemExit(f"{failures} of {attempts} docstring examples failed")
    print(
        f"Executed {blocks} guide examples, {len(gallery.EXAMPLES)} gallery plots, "
        f"{rendered_attempts} rendered doctest checks, and {attempts} source doctest checks."
    )
    import matplotlib.pyplot as plt

    plt.close("all")


if __name__ == "__main__":
    main()
