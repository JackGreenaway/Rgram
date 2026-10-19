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

## Writing documentation

Documentation pages are Markdown files in `docs/`. Keep the same division used
by scikit-learn: the **user guide** teaches concepts and practical choices, the
**API reference** specifies exact object behavior, and the **examples gallery**
shows complete workflows. A reader should be able to move from an explanation
to a runnable example and then to the exact API contract without finding the
same long explanation copied in several places. See scikit-learn's
[documentation guidance](https://scikit-learn.org/stable/developers/contributing.html#documentation)
and [user guide](https://scikit-learn.org/stable/user_guide.html) for the model.

| Page type | Organize the explanation this way | Source of truth |
| --- | --- | --- |
| Getting started | State the task, show the shortest complete fit/predict workflow, explain the output, then link to the next relevant guide. | `docs/getting_started.md` |
| User guide | Begin with the question or choice; explain the idea and assumptions; show a focused example; interpret its output; describe limits and alternatives. Link exact parameters to the API and link longer workflows to the gallery. | `docs/*.md` |
| Example gallery | Give the practical question and data context; state what is held fixed and what changes; show the figure near the explanation; provide complete runnable code on the page; explain how to read it, when to use the approach, its limits, and a follow-up. | `examples/plot_relationships.py` and `examples/held_out_prediction.py` |
| API reference | Lead with a concise purpose statement; document every parameter, return value, fitted attribute, warning, and relevant exception; keep examples short and executable. | NumPy-style docstrings in `src/rgram/` |
| Development and implementation | Explain contributor actions and internal design. Link public behavior back to the guide or API instead of making this the only description of a user-facing feature. | `docs/development.md` and `docs/internals.md` |

Keep pages focused on one topic and use one `#` title per page. Use `##` for
main sections and `###` only when a section needs real subdivisions. Prefer
question-led headings and connected prose over a sequence of parameter tables;
use a table when readers need to compare choices or scan a contract. Put API
names, parameter names, and file paths in backticks. Link related pages with
relative Markdown paths, and link public API names to their reference pages.
Explain statistical assumptions, units, tradeoffs, and limits beside the
recommendation or result they qualify. Use admonitions only for information a
reader should notice before acting, and give every figure useful alt text.

The site keeps scikit-learn's familiar three-part layout: topic navigation at
the left, the page content in the center, and section links at the right. The
top navigation stays limited to the main documentation areas. Preserve this
hierarchy when adding pages: add the file to the appropriate group in
`mkdocs.yml`, use existing components such as `.docs-gallery` for collections
of examples, and let figures scale within the content column. Avoid page-specific
layout or styling when a shared component can express the same structure.

Fence runnable snippets with `python`, shell commands with `bash`, and doctests
with `pycon`. Keep examples focused on the behavior being explained, but include
the imports and deterministic data needed to run them. Python blocks on one page
share a namespace during validation; each page starts fresh. If output is shown,
put a `text` block immediately after its Python block so the checker can compare
it. Avoid network access and interactive input because examples run headlessly.
Edit gallery functions in `examples/plot_relationships.py`, then regenerate the
page code, downloads, and figures. Do not hand-edit generated gallery blocks or
content between `BEGIN REFERENCE` and `END REFERENCE` markers.

Before submitting a documentation change, run the checks in
[Build and validate](#build-and-validate), regenerating references or gallery
assets when their sources changed. Preview the site with `mkdocs serve` and check
the edited page's navigation, code formatting, equations, and images. Commit the
Markdown and any regenerated assets together.

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
