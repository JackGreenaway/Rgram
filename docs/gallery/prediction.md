# Predicting demand from temperature

## Question and data

Can a nonlinear temperature curve predict demand on unseen days better than a
constant training mean? We generate 160 independent synthetic days with a
U-shaped mean and Gaussian noise. This controlled example needs no download;
it does not establish how real demand behaves.

## Configuration

Hold out 25% of days before selecting a bandwidth. `GridSearchCV` compares three
Gaussian bandwidths in a `Pipeline`: each fold learns its own `StandardScaler`.
The test set influences neither scaling nor selection. The bandwidth is measured
in standard deviations of training temperature. Gaussian support permits finite
predictions throughout this dataset; extrapolation is explicitly allowed and
warned about, rather than being evidence of accuracy outside the training range.

Random splitting assumes independent days. Use chronological or grouped splits
when observations are dependent. Unsorted-input warnings are expected because
splitting preserves randomly selected rows rather than sorting them.

Download [the standalone source](../_static/examples/held_out_prediction.py).

## Run the complete example

```bash
python examples/held_out_prediction.py
```

```python
"""Can a temperature curve predict demand for unseen observations?"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rgram import KernelSmoother

# Independent synthetic days; no network access or hidden data cleaning.
rng = np.random.default_rng(42)
temperature = rng.uniform(5, 30, 160)
demand = 20 + 0.15 * (temperature - 18) ** 2 + rng.normal(0, 2, 160)
# One column is required by sklearn preprocessing; rows are independent days.
X = temperature.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    demand,
    test_size=0.25,
    random_state=42,
)
# The scaler learns only from each training fold; bandwidth uses scaled units.
pipeline = make_pipeline(
    StandardScaler(),
    KernelSmoother(kernel="gaussian", bandwidth="manual", extrapolation="allow"),
)
search = GridSearchCV(
    pipeline,
    {"kernelsmoother__bandwidth_value": [0.1, 0.3, 0.7]},
    scoring="neg_mean_squared_error",
    cv=4,
).fit(X_train, y_train)
prediction = search.predict(X_test)
print(
    f"Selected bandwidth: {search.best_params_['kernelsmoother__bandwidth_value']:.1f}"
)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, prediction)):.2f}")
print(
    f"Training-mean baseline RMSE: {np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))):.2f}"
)
```

```text
Selected bandwidth: 0.1
Test RMSE: 2.11
Training-mean baseline RMSE: 7.14
```

## Interpretation and next steps

RMSE is in demand units. On this seeded dataset, the selected smoother improves
on the constant baseline. These numbers describe this simulation and split;
they are not a performance guarantee. Selection uses negative MSE, so larger
CV scores mean less error. The final test RMSE is evaluated only once.

Next inspect held-out residuals and support, repeat with appropriate resampling,
and compare a domain baseline. Do not tune further using this test set and still
report it as an untouched evaluation. See [pipelines](../sklearn.md) and
[theory](../theory.md) for assumptions and limitations.
