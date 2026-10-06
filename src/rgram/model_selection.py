"""Model selection that never silently rewards missing predictions."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils.validation import check_is_fitted

from rgram._typing import CV, Frame, Input, Prediction
from rgram.base import BaseUtils


class CoverageSearchCV(RegressorMixin, BaseEstimator):
    """Select parameters by validation MSE with explicit support requirements.

    Parameters
    ----------
    estimator : sklearn estimator
        Cloneable single-feature regressor with fit and predict. Fit errors and
        prediction policies are respected, not overridden.
    param_grid : dict or list of dicts
        Parameter names mapped to sequences of candidates, as in ParameterGrid.
        Nested names use the usual step__parameter convention.
    cv : int, splitter or iterable, default=5
        Integer uses shuffled KFold. Otherwise supply a split(X, y, groups)
        object or iterable of nonempty, disjoint train/test index pairs.
    min_coverage : float, default=1.0
        Minimum fraction of finite predictions required in every fold, in (0, 1].
        Lowering this compares only supported rows and can favor selective models.
    random_state : int or None, default=0
        Seed for integer-CV splitting. Does not reseed a supplied splitter.

    Attributes
    ----------
    best_estimator_ : estimator
        Winning model refitted on all rows. Available only after successful fit.
    best_params_ : dict
        Winning parameter combination.
    best_index_ : int
        Position of the winner in ``cv_results_``. Ties use the first candidate.
    best_score_ : float
        Negative pooled unweighted validation MSE, not R-squared.
    cv_results_ : dict
        Candidate parameters, MSE, support fractions, eligibility, selection loss,
        validation counts, and per-fold MSE/coverage arrays. See result schemas.
    cv_splits_ : tuple of (train_indices, test_indices)
        Copies of the actual fold membership indices.
    n_splits_ : int
        Number of supplied validation folds.
    n_features_in_ : int
        Always one after successful fit.
    feature_names_in_ : ndarray of shape (1,)
        Original feature name, only when the training input is named.

    Notes
    -----
    The selection loss pools squared errors over finite validation predictions.
    Each fold must independently meet min_coverage. Ineligible candidates have
    infinite selection loss. If none qualifies, fit raises and ``cv_results_`` remains
    inspectable, but the search is not fitted. There is no n_jobs, scoring, refit,
    or error_score option. Inherited score reports R-squared.
    Weights affect fitting only. Direct weight forwarding to nested pipelines is
    not implemented. Unlike GridSearchCV's integer regression CV, integer CV here
    is shuffled. Preprocessing must be inside the candidate pipeline to avoid leakage.

    Examples
    --------
    >>> import numpy as np
    >>> from rgram import CoverageSearchCV, Regressogram
    >>> X = np.linspace(0, 1, 30).reshape(-1, 1)
    >>> search = CoverageSearchCV(Regressogram(), {'n_bins': [1, 3]}, cv=3)
    >>> _ = search.fit(X, X[:, 0])
    >>> search.predict(X).shape
    (30,)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Union[dict[str, Sequence[Any]], list[dict[str, Sequence[Any]]]],
        *,
        cv: CV = 5,
        min_coverage: float = 1.0,
        random_state: Optional[int] = 0,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.min_coverage = min_coverage
        self.random_state = random_state

    def fit(
        self,
        X: Input = None,
        y: Input = None,
        *,
        x: Input = None,
        data: Optional[Frame] = None,
        groups: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
    ) -> CoverageSearchCV:
        """Evaluate candidates and refit the eligible winner on all observations.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1), or str, default=None
            Single feature or its column name when data is supplied.
        y : array-like or str, default=None
            Aligned numeric response or its column name. Required.
        x : array-like or str, default=None
            Legacy alias for X; supply exactly one alias.
        data : polars.DataFrame or polars.LazyFrame, default=None
            Source for named feature and response columns.
        groups : array-like of shape (n_samples,) or None, default=None
            Labels passed to an explicit group-aware splitter. Rejected with integer cv.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Nonnegative fitting weights, sliced by training fold. Scoring is unweighted.

        Returns
        -------
        self : CoverageSearchCV
            Fitted search. The winner is refit on every supplied row.

        Raises
        ------
        ValueError
            For invalid input, fold indices, policies, or when no candidate qualifies.
            Candidate fit/prediction errors propagate rather than becoming scores.
        """
        x = BaseUtils._resolve_X(X, x)
        for name in (
            "best_estimator_",
            "best_params_",
            "best_index_",
            "best_score_",
            "cv_results_",
        ):
            self.__dict__.pop(name, None)
        if (
            isinstance(self.min_coverage, (bool, str))
            or not np.isscalar(self.min_coverage)
            or not np.isfinite(self.min_coverage)
            or not 0 < self.min_coverage <= 1
        ):
            raise ValueError("min_coverage must be in (0, 1]")
        feature_name = BaseUtils._feature_name(x, data)
        self.__dict__.pop("feature_names_in_", None)
        frame, _, _ = BaseUtils()._prepare_data(x, y, data)
        frame = frame.collect()
        X = frame["x"].to_numpy().reshape(-1, 1)
        y = frame["y"].to_numpy()
        if groups is not None and len(groups) != len(y):
            raise ValueError("groups length must match training data")
        if sample_weight is not None:
            sample_weight = BaseUtils._prediction_array(sample_weight)
            if (
                len(sample_weight) != len(y)
                or (sample_weight < 0).any()
                or not (sample_weight > 0).any()
            ):
                raise ValueError(
                    "sample_weight must match rows and have nonnegative, positive total mass"
                )
        if isinstance(self.cv, (int, np.integer)) and not isinstance(self.cv, bool):
            if groups is not None:
                raise ValueError(
                    "groups supplied with integer cv; use an explicit group-aware splitter such as GroupKFold"
                )
            splits = KFold(self.cv, shuffle=True, random_state=self.random_state).split(
                X, y
            )
        elif hasattr(self.cv, "split"):
            splits = self.cv.split(X, y, groups)
        else:
            splits = iter(self.cv)
        folds = []
        for train, test in splits:
            train, test = np.asarray(train), np.asarray(test)
            for indices in (train, test):
                if (
                    indices.ndim != 1
                    or not np.issubdtype(indices.dtype, np.integer)
                    or not len(indices)
                ):
                    raise ValueError(
                        "CV folds must contain nonempty 1D integer indices"
                    )
                if (
                    (indices < 0).any()
                    or (indices >= len(y)).any()
                    or len(np.unique(indices)) != len(indices)
                ):
                    raise ValueError("CV fold indices must be unique and in range")
            if np.intersect1d(train, test).size:
                raise ValueError("CV training and validation rows must not overlap")
            folds.append((train.copy(), test.copy()))
        if not folds:
            raise ValueError("cv must provide at least one split")
        results = {
            "params": [],
            "mean_test_mse": [],
            "coverage": [],
            "min_fold_coverage": [],
            "eligible": [],
            "selection_loss": [],
            "n_validation": [],
            "n_supported": [],
        }
        for index in range(len(folds)):
            results[f"split{index}_coverage"] = []
            results[f"split{index}_test_mse"] = []
        self.cv_splits_ = tuple((train.copy(), test.copy()) for train, test in folds)
        for parameters in ParameterGrid(self.param_grid):
            errors, covered, total, fold_coverages = 0.0, 0, 0, []
            for index, (train, test) in enumerate(folds):
                model = clone(self.estimator).set_params(**parameters)
                # User-selected error, support and warning policies are respected.
                fit_kw = (
                    {}
                    if sample_weight is None
                    else {"sample_weight": sample_weight[train]}
                )
                model.fit(X[train], y[train], **fit_kw)
                prediction = np.asarray(model.predict(X[test]))
                if prediction.shape != y[test].shape:
                    raise ValueError(
                        "Estimator must return one prediction per validation row"
                    )
                valid = np.isfinite(prediction)
                squared = np.square(prediction[valid] - y[test][valid])
                loss = float(squared.mean()) if valid.any() else np.inf
                coverage = float(valid.mean())
                results[f"split{index}_coverage"].append(coverage)
                results[f"split{index}_test_mse"].append(loss)
                fold_coverages.append(coverage)
                errors += squared.sum()
                covered += valid.sum()
                total += len(test)
            mse = float(errors / covered) if covered else np.inf
            eligible = min(fold_coverages) >= self.min_coverage and np.isfinite(mse)
            values = (
                parameters,
                mse,
                covered / total,
                min(fold_coverages),
                eligible,
                mse if eligible else np.inf,
                total,
                int(covered),
            )
            for key, value in zip(
                (
                    "params",
                    "mean_test_mse",
                    "coverage",
                    "min_fold_coverage",
                    "eligible",
                    "selection_loss",
                    "n_validation",
                    "n_supported",
                ),
                values,
            ):
                results[key].append(value)
        self.cv_results_ = {
            k: v if k == "params" else np.asarray(v) for k, v in results.items()
        }
        self.n_splits_ = len(folds)
        self.cv_splits_ = tuple((train.copy(), test.copy()) for train, test in folds)
        if not np.isfinite(self.cv_results_["selection_loss"]).any():
            raise ValueError(
                "No candidate meets min_coverage in every fold; inspect cv_results_ or broaden support"
            )
        self.best_index_ = int(np.argmin(self.cv_results_["selection_loss"]))
        self.best_params_ = results["params"][self.best_index_]
        self.best_score_ = -results["selection_loss"][self.best_index_]
        best = clone(self.estimator).set_params(**self.best_params_)
        fit_kw = {} if sample_weight is None else {"sample_weight": sample_weight}
        final_X = (
            pl.DataFrame({feature_name: X[:, 0]}) if feature_name is not None else X
        )
        best.fit(final_X, y, **fit_kw)
        if feature_name is not None:
            self.feature_names_in_ = np.asarray([feature_name], dtype=object)
        self.best_estimator_ = best
        self.n_features_in_ = 1
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Require a refitted winner; retained results alone do not make an unsuccessful search fitted."""
        return hasattr(self, "best_estimator_")

    def predict(self, X: Input = None, *, x: Input = None) -> Prediction:
        """Predict through the refitted winning estimator.

        Parameters
        ----------
        X : array-like of shape (n_queries,) or (n_queries, 1), default=None
            Query feature values, with the fitted feature name if named.
        x : array-like, default=None
            Legacy alias for X; supply exactly one alias.

        Returns
        -------
        prediction : ndarray of shape (n_queries,)
            Winner's predictions in query order, respecting its prediction policies.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If a successful search has not produced ``best_estimator_``.
        """
        x = BaseUtils._resolve_X(X, x)
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(x)
