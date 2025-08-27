import polars as pl
import polars_ols as pls  # noqa: F401

from src.rgram.base import BaseUtils

from typing import Sequence, cast, List


class KernelSmoother(BaseUtils):
    """
    KernelSmoother

    Epanechnikov kernel regression smoother for one-dimensional data.

    Parameters
    ----------
    data : pl.DataFrame or pl.LazyFrame
        Input data.
    x : str
        Feature column.
    y : str
        Target column.
    hue : sequence of str, optional
        Optional grouping variable(s).
    n_eval_samples : int, default=100
        Number of evaluation points for the smoother.

    Methods
    -------
    fit()
        Fit the kernel smoother to the data.
    transform()
        Return the kernel smoothed results after fitting.
    fit_transform()
        Fit to data, then return the kernel smoothed results.
    """

    def __init__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        x: str,
        y: str,
        hue: Sequence[str] | None = None,
        n_eval_samples: int = 100,
    ) -> None:
        """
        Construct a KernelSmoother instance.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input data.
        x : str
            Feature column.
        y : str
            Target column.
        hue : sequence of str, optional
            Optional grouping variable(s).
        n_eval_samples : int, default=100
            Number of evaluation points for the smoother.
        """
        super().__init__(hue=hue)

        self.data = data.lazy()
        self.x: list[str] = cast(List[str], self._to_list(x))
        self.y: list[str] = cast(List[str], self._to_list(y))
        # self.hue: List[str] = cast(List[str], self._to_list(hue) or [])
        self.n_eval_samples = n_eval_samples

    def _calculate_bandwidth(self) -> pl.Expr:
        """
        Calculate the kernel bandwidth using Silverman's rule of thumb.

        Returns
        -------
        pl.Expr
            The bandwidth expression.
        """
        # Compute std and IQR only once for efficiency
        std_expr = pl.col(self.x).std()
        iqr_expr = (
            pl.col(self.x).quantile(0.75) - pl.col(self.x).quantile(0.25)
        ) / 1.34
        bw = self._over_function(
            0.9 * pl.min_horizontal([std_expr, iqr_expr]) * (pl.len() ** (-1 / 5))
        ).alias("h")

        return bw

    def _calculate_x_eval(self) -> pl.Expr:
        """
        Calculate the evaluation points for the kernel smoother.

        Returns
        -------
        pl.Expr
            The evaluation points expression.
        """
        x_eval = self._over_function(
            pl.linear_spaces(
                pl.col(self.x).min(),
                pl.col(self.x).max(),
                self.n_eval_samples,
                as_array=True,
            )
        ).alias("x_eval")

        return x_eval

    def fit(self) -> "KernelSmoother":
        """
        Fit the kernel smoother to the data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        bw = self._calculate_bandwidth()
        x_eval = self._calculate_x_eval()

        ks = (
            self.data.with_columns([bw, x_eval])
            .explode("x_eval")
            .with_columns(
                [
                    ((pl.col("x_eval") - pl.col(self.x)) / pl.col("h")).alias("u"),
                ]
            )
            .with_columns(
                [
                    (0.75 * (1 - (pl.col("u") ** 2))).alias("weight"),
                ]
            )
            .filter(pl.col("u").abs() <= 1)
            .group_by(["x_eval"] + self.hue)
            .agg(
                [
                    # Epanechnikov kernel
                    (
                        (pl.col(self.y) * pl.col("weight")).sum()
                        / pl.col("weight").sum()
                    ).alias("y_kernel")
                ]
            )
            .sort(by="x_eval")
        )

        self._ks_result = ks

        return self

    def transform(self) -> pl.LazyFrame:
        """
        Return the kernel smoothed results after fitting.

        Returns
        -------
        pl.LazyFrame
            The kernel smoothed results.
        """
        if not hasattr(self, "_ks_result"):
            raise RuntimeError("You must call fit() before transform().")

        return self._ks_result

    def fit_transform(self) -> pl.LazyFrame:
        """
        Fit to data, then return the kernel smoothed results.

        Returns
        -------
        pl.LazyFrame
            The kernel smoothed results.
        """
        self.fit()

        return self.transform()
