# API reference

## API stability

Rgram has an **unstable API**. Parameters, methods, and result schemas may change
between releases without backward compatibility or a deprecation period.
Pin the package version for reproducible analyses.


All top-level exports are documented below. The estimators also inherit
scikit-learn parameter, scoring, and metadata-request methods. Metadata-request
methods are included for completeness; their presence does not mean Rgram
implements full metadata routing, especially through `CoverageSearchCV` pipelines.
Use the explicit weight forwarding described in the [sklearn guide](../sklearn.md).

- [Regressogram](regressogram.md)
- [Kernel Smoother](kernel_smoother.md)
- [Coverage Search](coverage_search.md)
- [Aggregation](aggregation.md)
- [Plotting](plotting.md)
- [Warnings](warnings.md)
- [Results](results.md)

## Public interface

| Object | Purpose |
|---|---|
| [Regressogram](regressogram.md) | Binned response summaries. |
| [KernelSmoother](kernel_smoother.md) | Local-constant and local-linear kernel regression. |
| [CoverageSearchCV](coverage_search.md) | Validation MSE search requiring explicit prediction coverage. |
| [rgram.array_aggregation](aggregation.md#rgram.array_aggregation) | Numeric per-bin scalar callback adapter. |
| [rgram.weighted_aggregation](aggregation.md#rgram.weighted_aggregation) | Native Polars weighted callback adapter. |
| [rgram.quantile](aggregation.md#rgram.quantile) | Explicit unweighted quantile reducer. |
| [plot_diagnostics](plotting.md) | Optional residual and observed-versus-predicted figure. |
| [Warning categories](warnings.md) | Filter support, ordering, extrapolation, binning, and numerical advisories. |

The helper factories return cloneable classes documented under
`rgram.aggregation`. Module internals are explained separately in the
[implementation guide](../internals.md).
