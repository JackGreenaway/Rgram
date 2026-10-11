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
