"""Filterable warnings emitted by rgram estimators."""


class RgramWarning(UserWarning):
    """Base category; filter this to silence all rgram advisory warnings."""


class UnsortedInputWarning(RgramWarning):
    """Input order is valid for estimation but may matter when drawing lines."""


class SupportWarning(RgramWarning):
    """Predictions or bootstrap intervals have insufficient support."""


class NumericalWarning(RgramWarning):
    """A local fit required an explicit numerical fallback."""


class BinningWarning(RgramWarning):
    """An automatic bin count was bounded to avoid excessive resolution."""


class ExtrapolationWarning(RgramWarning):
    """A prediction query lies outside the fitted training range."""


class DataHandlingWarning(RgramWarning):
    """A numerical rule used a documented fallback or changed resolution."""
