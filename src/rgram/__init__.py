"""
Rgram: High-performance nonparametric regression library.

A Python library for regressograms and kernel smoothing, built on Polars for fast
data processing. Provides binned regression estimation and local kernel
smoothing for univariate data with optional confidence intervals.

Classes
-------
Regressogram
    Binned regression estimator with multiple binning strategies and customizable aggregation.
KernelSmoother
    Local-constant/linear kernel regression with bandwidth and influence controls.
CoverageSearchCV
    Validation-MSE grid search with explicit prediction-coverage requirements.

References
----------
García-Portugués, E. (2023). Notes for nonparametric statistics.
Carlos III University of Madrid.
"""

from .model_selection import CoverageSearchCV
from .rgram import Regressogram
from .smoothing import KernelSmoother
from .warnings import (
    BinningWarning,
    NumericalWarning,
    RgramWarning,
    SupportWarning,
    UnsortedInputWarning,
)

__all__ = [
    "Regressogram",
    "KernelSmoother",
    "CoverageSearchCV",
    "RgramWarning",
    "UnsortedInputWarning",
    "SupportWarning",
    "NumericalWarning",
    "BinningWarning",
]

from .aggregation import array_aggregation, quantile, weighted_aggregation
from .plotting import plot_diagnostics
from .warnings import DataHandlingWarning, ExtrapolationWarning

__all__ += [
    "array_aggregation",
    "weighted_aggregation",
    "quantile",
    "plot_diagnostics",
    "ExtrapolationWarning",
    "DataHandlingWarning",
]
