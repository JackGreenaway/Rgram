# Development and publishing documentation

```{toctree}
:maxdepth: 2

internals
```

## Library development

From the repository root:

```bash
uv sync
uv run pytest -q
uv run ruff check src/rgram examples tests/test_estimator_contract.py tests/test_advanced_estimators.py tests/test_aggregation_and_transparency.py tests/test_properties.py
uv build
```

Tests cover estimator integration, numerical references, aggregation, weighted
fits, validation, row preservation, search coverage, intervals, and plotting
callbacks. The package metadata and runtime dependencies are in `pyproject.toml`;
`uv.lock` records the development environment. CI tests Python 3.9 and 3.12.

## Build the documentation

The site uses Sphinx, MyST Markdown, NumPy-style docstrings, and the PyData Sphinx
theme. Scikit-learn also uses [PyData in its documentation configuration](https://github.com/scikit-learn/scikit-learn/blob/main/doc/conf.py).
The documentation dependencies are separate from runtime dependencies.

```bash
python -m pip install -e '.[plot]' -r docs/requirements.txt
python scripts/check_docs.py
python scripts/check_examples.py
python -m sphinx -b html -W --keep-going docs docs/_build/html
python scripts/check_docs.py --html docs/_build/html
python -m http.server --directory docs/_build/html 8000
```

Open `http://localhost:8000`. The output is a static site with search, navigation,
API signatures, source links, and MathJax equations. MathJax is loaded from a CDN;
viewing typeset equations requires network access. Site assets and internal links
use relative paths so a GitHub Pages repository subpath works.

The coverage check enumerates top-level exports, public estimator methods,
constructor/method arguments, and package modules. The HTML check verifies API
anchors and all local page, fragment, and asset links. The example check executes
all guide snippets and public docstring examples headlessly. The strict Sphinx
build treats warnings as errors. External source availability can be checked separately
with `python -m sphinx -b linkcheck docs docs/_build/linkcheck`; transient external
network failures do not block the normal HTML build.

## Publish to GitHub Pages

The repository includes `.github/workflows/docs.yml`. It builds and validates the
site on relevant pushes and pull requests, and offers a manual run. Deployment
runs only on the repository's default branch, never on pull requests.

1. In the repository's **Settings → Pages**, set **Source** to **GitHub Actions**.
2. Commit and push the documentation changes to your default branch, or run the
   **Documentation** workflow on that branch.
3. The workflow uploads `docs/_build/html` and deploys to the `github-pages`
   environment. Use the URL shown in the deployment job.

The Sphinx GitHub Pages extension emits `.nojekyll`, preserving directories such
as `_static`. The workflow follows [GitHub's custom workflow requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
No generated HTML needs to be committed and no third-party hosting service is
required. This setup prepares publication; a local build does not publish a site.

## Keep the reference complete

When adding an export or public method, add its API directive and document its
parameters, returned objects, errors, and behavior. Update fitted-attribute and
result-schema tables when adding outputs. Explain new statistical assumptions in
the user guide, and add an example when they change how users interpret a result.

Internal modules and helper responsibilities are listed in
[the implementation guide](internals.md). Private helpers are implementation
details, not additional supported user APIs. API pages obtain signatures and
method documentation directly from the installed source through
[Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).
