"""Execute the documentation's Python examples and public API doctests headlessly."""

from __future__ import annotations

from contextlib import redirect_stdout
import doctest
import importlib
from io import StringIO
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


def main():
    pages = ["getting_started.md", "sklearn.md", "examples.md", "api/warnings.md"]
    blocks = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in pages:
            namespace = {}
            text = (ROOT / "docs" / name).read_text()
            for code in re.findall(r"```python\n(.*?)```", text, re.S):
                # Keep figures/tables out of CI logs; failures still propagate.
                with redirect_stdout(StringIO()):
                    exec(compile(code, f"docs/{name}", "exec"), namespace)
                blocks += 1
        failures = attempts = 0
        for module in (
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
        f"Executed {blocks} guide examples and {attempts} docstring examples successfully."
    )
    import matplotlib.pyplot as plt

    plt.close("all")


if __name__ == "__main__":
    main()
