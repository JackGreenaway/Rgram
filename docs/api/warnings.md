# Warning categories

Advisory warnings use Python's standard filters and do not change a fit or its
error policies. All categories below inherit from `RgramWarning`, which inherits
from `UserWarning`.

| Category | When it occurs |
|---|---|
| `UnsortedInputWarning` | Training/query feature order is not nondecreasing. Rows remain in their original order. |
| `SupportWarning` | Query estimates or bootstrap draws have inadequate support and the selected policy permits NaN. |
| `NumericalWarning` | A singular/unresolved local-linear fit uses a local-constant fallback. |
| `BinningWarning` | Automatic count capped, an explicit count reduced, or repeated boundaries merged. |
| `ExtrapolationWarning` | A query is outside the training feature range and the policy permits evaluation or NaN. |
| `DataHandlingWarning` | Silverman's zero-IQR rule uses a positive standard deviation fallback. |

```python
import warnings
from rgram import UnsortedInputWarning, RgramWarning

# A specific advisory, for a workflow where feature order is irrelevant:
warnings.filterwarnings("ignore", category=UnsortedInputWarning)
# Or apply this inside a catch_warnings() block to silence all Rgram advisories:
warnings.filterwarnings("ignore", category=RgramWarning)
```

Feature-name compatibility warnings are standard `UserWarning`, matching
scikit-learn's conventions, and are not suppressed by filtering `RgramWarning`.
Validation errors and strict support/extrapolation errors remain active. Python
can also escalate warnings to exceptions with `filterwarnings("error", ...)`.

```{eval-rst}
.. autoexception:: rgram.RgramWarning
```
```{eval-rst}
.. autoexception:: rgram.UnsortedInputWarning
```
```{eval-rst}
.. autoexception:: rgram.SupportWarning
```
```{eval-rst}
.. autoexception:: rgram.NumericalWarning
```
```{eval-rst}
.. autoexception:: rgram.BinningWarning
```
```{eval-rst}
.. autoexception:: rgram.ExtrapolationWarning
```
```{eval-rst}
.. autoexception:: rgram.DataHandlingWarning
```

Implementation source: {download}`warnings.py <../../src/rgram/warnings.py>`.
