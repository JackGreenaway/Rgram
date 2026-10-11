# Development and GitHub Pages

Rgram reuses its MkDocs runtime and custom theme. Its organization follows
scikit-learn's [user guide](https://scikit-learn.org/stable/user_guide.html),
[examples gallery](https://scikit-learn.org/stable/auto_examples/index.html),
[class reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
and [function reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
inspected on 11 October 2026: concepts in the guide, complete question-led
examples, and concise generated API contracts with source links and doctests. Documentation
and development dependencies are recorded in `uv.lock`, separately from runtime
requirements. Tests cover Python 3.9 and 3.12; documentation and releases use 3.11.

## Environment setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the
repository, and run the commands below from its root. `uv sync` installs the
package in editable mode and the default development group; `--group docs` adds
the documentation tools. It can install Python 3.11 if needed.

Use the locked Python 3.11 environment for reference generation and image checks:
inherited scikit-learn docstrings and rendering dependencies differ between
Python versions. Runtime tests can also run on Python 3.9 and 3.12. No additional
documentation framework or runtime dependency is required.

## Build and validate

```bash
uv sync --locked --group docs --python 3.11
uv run --locked --python 3.11 --group docs python scripts/build_api_reference.py
uv run --locked --python 3.11 --group docs python scripts/build_gallery.py
uv run --locked --python 3.11 --group docs python scripts/build_gallery.py --check
uv run --locked --python 3.11 --group docs python scripts/build_api_reference.py --check
uv run --locked --python 3.11 --group docs python scripts/check_docs.py
uv run --locked --python 3.11 --group docs python scripts/check_examples.py
uv run --locked --python 3.11 --group docs mkdocs build --strict
uv run --locked --python 3.11 --group docs python scripts/check_docs.py --html site
uv run --locked --python 3.11 --group docs mkdocs serve
```

Open `http://127.0.0.1:8000`. The site includes search, dark/light mode,
responsive navigation, source links, and MathJax equations. Equations require
access to the MathJax CDN. Generated HTML lives in `site/` and is ignored by Git.

## Keep references aligned with source

Reference blocks are generated from live signatures and NumPy-style docstrings,
including public inherited estimator methods. After changing code or docstrings,
run `scripts/build_api_reference.py` and commit the updated Markdown. CI rejects
stale reference blocks. Add reference markers for new exports, document their
arguments and fitted state, and update [result schemas](api/results.md).

The coverage check enumerates exports, methods, parameters, attributes, and
modules. The HTML check verifies API anchors and local page, fragment, and asset
links. The example check runs guide snippets, public doctests, and gallery plots
headlessly, including every Markdown Python block and displayed output paired
with a Python block. Generated `pycon` blocks are checked with doctest too.
Gallery code and downloads are derived from `examples/plot_relationships.py`;
run `build_gallery.py` after editing those functions. Regenerate plot images with
`MPLBACKEND=Agg uv run python examples/plot_relationships.py --output-dir docs/_static/gallery`.
The prediction source, download, and page code are also synchronized by
`build_gallery.py`. The output check catches result drift; rendered gallery PNGs
are compared pixel for pixel with the checked-in images. MkDocs treats navigation and link warnings as errors.

See [implementation details](internals.md) for contributor guidance.

## Publish GitHub Pages

1. Set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**.
2. Push to `main`, or manually run the **Documentation** workflow on `main`.
3. The build validates the site and uploads `site/`; the deploy job publishes it
   to the `github-pages` environment. Pull requests only build and validate.

The published URL is <https://jackgreenaway.github.io/Rgram/>. A local build does
not publish the site. If raw Markdown appears, check that Pages uses GitHub
Actions and that both workflow jobs succeeded.

## Tests and releases

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check src/rgram examples tests/test_estimator_contract.py tests/test_advanced_estimators.py tests/test_aggregation_and_transparency.py tests/test_properties.py scripts
uv build
```

The **Create GitHub Release** workflow validates the package, reads
`project.version`, rejects existing version tags, checks wheel/source distributions,
and creates `vX.Y.Z` with generated release notes. Run it from the intended release
commit. The **Upload Python Package** workflow publishes those validated artifacts
after a successful release workflow, or builds and validates a release published
through GitHub's UI. Published tags must match `project.version`.

Configure a PyPI trusted publisher for `JackGreenaway/Rgram`, workflow
`python-publish.yml`, environment `pypi`. Publishing uses OpenID Connect. Repository
Pages and PyPI settings must be configured before their first deployment.
