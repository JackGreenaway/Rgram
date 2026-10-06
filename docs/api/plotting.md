# Plotting

```{eval-rst}
.. autofunction:: rgram.plot_diagnostics
```

The helper expects a Rgram estimator and feature values in that estimator's
coordinate system. For a pipeline, transform the features through
`pipeline[:-1]` and pass its final estimator. It returns a Matplotlib figure and
a two-element array of axes; calling `plt.show()` or saving the figure is up to
the caller. See the [worked examples](../examples.md).

Implementation source: {download}`plotting.py <../../src/rgram/plotting.py>`.
