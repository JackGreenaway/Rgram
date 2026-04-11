"""
Rgram: High-performance nonparametric regression library.

A Python library for regressograms and kernel smoothing, built on Polars for fast
data processing. Provides binned regression estimation and Epanechnikov kernel
smoothing for univariate data with optional confidence intervals.

Classes
-------
Regressogram
    Binned regression estimator with multiple binning strategies and customizable aggregation.
KernelSmoother
    Epanechnikov kernel regression smoother with adaptive bandwidth selection.

References
----------
García-Portugués, E. (2023). Notes for nonparametric statistics. 
Carlos III University of Madrid.
"""

from .rgram import Regressogram
from .smoothing import KernelSmoother


__all__ = ["Regressogram", "KernelSmoother"]
