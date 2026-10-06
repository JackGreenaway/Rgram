# CoverageSearchCV

This optional search object is deliberately narrower than `GridSearchCV`.
It uses pooled unweighted validation MSE and rejects candidates with inadequate
support in any fold. See [search result fields](results.md#search-results) and
[the search contracts](../advanced.md#optional-cv-fit-weights-and-inspectable-splits).

```{eval-rst}
.. autoclass:: rgram.CoverageSearchCV
   :members:
   :inherited-members:
   :show-inheritance:
```

Implementation source: {download}`model_selection.py <../../src/rgram/model_selection.py>`.
