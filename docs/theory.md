# Theory, suitability, and interpretation

Rgram is an exploratory tool for **one numeric feature and one numeric response**.
Its curves summarize how the response behaves near feature values. The
recommendations below connect established nonparametric regression theory to
Rgram's implemented estimators; they are guidance for this library, rather than
claims that every nonparametric method has the same limitations.

## What relationship is being estimated?

For a mean response curve, the population quantity is

$$
m(x) = \mathbb{E}[Y\mid X=x].
$$

This averages the response conditional on one feature. It is neither the full
conditional distribution nor a density estimate of $X$. Fitting $Y$ against $X$
also answers a different question from fitting $X$ against $Y$.
See the [Statsmodels kernel regression reference](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html) for the conditional-mean target.

Changing `Regressogram.agg` changes the summary: a median or quantile describes
that statistic within a bin; a sum describes the observed bin total. A sum is
sensitive to occupancy and is not a conditional mean. Rgram's kernel regression
uses a local mean or a local least-squares line, not kernel quantile regression.

Unequal observation weights describe reweighted data. Depending on how weights
are chosen, the resulting curve may target a different population summary from
the unweighted conditional mean. Survey or inverse-probability interpretations
require a sampling/weighting model beyond Rgram's numerical influence contract.

## Regressograms: fixed cells and local summaries

Let $B_j$ be a fitted feature bin and $a_i$ the nonnegative observation weight.
With mean aggregation, Rgram evaluates

$$
\widehat m(x) =
\frac{\sum_{i:X_i\in B_j} a_iY_i}
     {\sum_{i:X_i\in B_j} a_i},\qquad x\in B_j,
$$

when the denominator is positive. With equal weights, this is the sample mean
of responses in that cell. The estimate is constant inside a bin and can jump
at boundaries. It describes a bin average, not variation resolved at every
point inside that bin. The [regressogram treatment](https://egarpor.github.io/NP-UC3M/kre-i.html#regressogram)
explains its connection to histograms and contrasts it with kernel regression.

Rgram offers equal-width and quantile-based cells, integer truncation, and exact
feature-value grouping. Quantile cells adapt their widths to observed feature
density; they do not guarantee equal counts when values tie. The fitted boundaries
are reused at prediction time. `bins_` shows occupied cells and counts; empty
cells can remain unsupported. See the [parameter reference](statistical_parameters.md)
for bin rules, boundary conventions, and weight behavior.

Use bins when a compact table of local summaries is more helpful than a smooth
curve. Do not infer a real discontinuity solely from a bin edge: moving the
boundaries or changing the number of bins can change the apparent step.

## Kernel regression: a moving neighborhood

For bandwidth $h>0$ and kernel $K$, define local weights

$$
w_i(x)=a_i K\!\left(\frac{X_i-x}{h}\right),\qquad
p_i(x)=\frac{w_i(x)}{\sum_j w_j(x)}.
$$

Local-constant regression evaluates $\widehat m(x)=\sum_i p_i(x)Y_i$ when there
is positive support. Unlike a regressogram, its neighborhood is centered on each
query. This is the Nadaraya–Watson estimator; multiplicative kernel normalization
constants cancel in the ratio. See the [kernel regression derivation](https://egarpor.github.io/NP-UC3M/kre-i.html#nadarayawatson-estimator).

A compact kernel assigns zero weight beyond its radius; it can leave gaps with
no estimate. Gaussian and logistic kernels have infinite mathematical support,
but a numerically finite estimate may depend almost entirely on very few rows.
Inspect `n_neighbors`, `effective_n`, and `get_weights`, rather than treating a
smooth-looking curve as evidence of abundant information.

## Local-linear regression and boundaries

At each query, local-linear regression solves

$$
(\widehat\beta_0,\widehat\beta_1)
=\operatorname*{arg\,min}_{\beta_0,\beta_1}
\sum_i w_i(x)\left[Y_i-\beta_0-\beta_1(X_i-x)\right]^2
$$

and reports $\widehat m(x)=\widehat\beta_0$. A local slope can reduce the boundary
bias of a local constant, where observations are available mostly on one side.
Its equivalent prediction coefficients can be negative, so predictions need not
stay inside the observed response range. This distinction is described in the
[Statsmodels kernel regression reference](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html).

In Rgram, an unresolved local slope triggers an explicit local-constant fallback
or an error, according to `singular`. This is reported in diagnostics. The
`local_linear` option does not impose a single global straight line.

Rgram's fixed bandwidth is measured in feature units. A tricube kernel with a
local line shares ideas with LOESS, but it is not an implementation of standard
LOESS: a nearest-neighbor span and iterative robust residual reweighting are not
implemented. See [NIST's LOESS description](https://itl.nist.gov/div898/handbook/pmd/section1/pmd144.htm).

## Smoothing trades resolution for stability

For a symmetric second-order kernel, independent random-design observations,
sufficiently smooth regression and density functions, and a supported interior
point, the leading kernel-regression bias is typically of order $h^2$ and
variance of order $1/(nh)$. Thus the squared-bias/variance balance has orders
$h^4$ and $1/(nh)$ and suggests an $n^{-1/5}$ bandwidth rate in that setting.
These statements require asymptotic regularity, including $h\to0$ and $nh\to\infty$;
they are not finite-sample error guarantees. Boundary behavior and sparse regions
need separate care. See [Predictive Modeling, nonparametric regression](https://egarpor.github.io/PM-UC3M/npreg.html).

Practically, compare a few bandwidths or bin counts. Very small neighborhoods can
follow noise; large ones can hide curvature or transitions. Rgram's Scott and
Silverman bandwidths are feature-scale rules of thumb, not regression-error
optimizers. Histogram bin rules likewise do not optimize response prediction.
Use explicit cross-validation if held-out prediction error is your criterion.

## Where Rgram is useful

| Question | Useful workflow | What to inspect |
|---|---|---|
| Does the average response bend, level off, or change direction? | Fit a mean regressogram or kernel curve over the observed feature range. | Several smoothing settings, raw observations, local support. |
| What are typical responses in understandable feature ranges? | Inspect `Regressogram.bins_`; use mean, median, or explicit quantiles. | Counts, weight totals, boundaries, and the meaning of the aggregation. |
| Do observed relationships differ between subgroups? | Explicitly select each subgroup and fit separate curves. | Overlapping feature ranges, group sizes, and common smoothing choices. |
| Is a global linear description missing structure? | Compare the exploratory curve and residuals with that description. | Remaining patterns; this is a diagnostic, not a formal lack-of-fit test. |
| How does a single feature predict a response? | Use a pipeline and held-out evaluation. | Validation error, coverage, and behavior outside the training range. |
| Which rows influence a particular fitted value? | Inspect `KernelSmoother.get_weights` on a small query set. | Concentration, zero support, and signed local-linear coefficients. |

Local regression is useful when a global functional form is unknown, but relies
on enough data around each location. NIST discusses both the flexibility and
local data demands of such methods in its [LOESS guide](https://itl.nist.gov/div898/handbook/pmd/section1/pmd144.htm).

## Where it is insufficient or inappropriate

| Intended conclusion or task | Why a Rgram curve is insufficient |
|---|---|
| A causal effect of changing a feature | Conditioning on one observed feature does not remove confounding or create an intervention. |
| An effect adjusted for other predictors, or an interaction | Separate pairwise fits do not condition jointly on several features. Multivariate estimation is intentionally outside scope. |
| A complete test of dependence | A flat mean can coexist with changing variance or other distributional changes. The curve does not provide a dependence-test p-value. |
| A definitive ranking of feature importance | Visual strength and in-sample fit depend on sample distribution and smoothing; they do not establish each feature's incremental value in a joint model. |
| Reliable prediction beyond observed feature support | Edge clipping or a numerical kernel estimate supplies a value without evidence for the relationship in that region. |
| Robust regression in the presence of extreme responses | Means and weighted least squares can be sensitive to outliers. A bin median changes the summary, but the smoother has no robust iterative fitting. |
| A classification, survival, or density-estimation API | Rgram supplies numeric response summaries; it has no classifier probabilities/calibration contract, censoring model, or density estimator. |
| A guaranteed monotone or bounded curve | The estimators impose neither monotonicity nor response bounds; local-linear values can overshoot. |
| Formal identification of a sharp threshold | Smoothing can blur a real jump, and bins can create apparent jumps. Dedicated change-point inference is a different task. |

These limits follow from the estimators and their implemented interfaces.
For example, with $Y=X\varepsilon$ and an independent zero-mean noise term,
$\mathbb{E}[Y\mid X]=0$ while conditional variance depends on $X^2$. A flat
mean curve therefore need not mean the variables are unrelated.

## Uncertainty, dependence, and evaluation

Rgram's `predict_interval` resamples paired feature/response/weight rows and refits
the curve. Percentile intervals take bootstrap quantiles; basic intervals reflect
those quantiles around the original estimate. These are pointwise curve intervals,
not future-observation intervals or simultaneous bands. See the [SciPy bootstrap
reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
for these interval definitions.

The procedure holds the selected bandwidth/bin count fixed, relearns bin boundaries
on each draw, and does not correct smoothing bias or include the selection process.
Paired bootstrap inference targets random-design sampling, rather than every
possible fixed-design regression experiment. Its adequacy depends on the data and
sampling process; more resamples only reduce Monte Carlo instability. Bootstrap
pitfalls and alternatives are discussed in [Nonparametric Statistics, regression
inference](https://egarpor.github.io/NP-UC3M/kre-ii.html#prediction-and-confidence-intervals).

IID resampling assumes independent rows. Whole-group and moving-block options
make different sampling assumptions; blocks use the supplied row order and need
a meaningful dependence structure and block length. Similarly, prediction
validation should respect subjects/groups and time rather than randomly mixing
dependent rows. See [scikit-learn's cross-validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

Unsupported bootstrap draws remain visible in `n_valid` and `bootstrap_coverage`.
Intervals default to NaN unless every draw supports a query. Lowering the valid
fraction conditions the reported bounds on supported draws. A narrow interval
cannot by itself validate an extrapolation or remove confounding.

## Further reading

The linked sources provide derivations and comparisons. Rgram-specific formulas,
defaults, and output definitions are in the [parameter reference](statistical_parameters.md)
and [API reference](api/index.md). The [worked examples](examples.md) show how to
inspect smoothing choices and subgroup relationships in practice.
