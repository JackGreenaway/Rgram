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

<!-- BEGIN REFERENCE:rgram.RgramWarning -->
<span id="rgram.RgramWarning"></span>

## rgram.RgramWarning



<div class="api-class-body" markdown="1">

Base category; filter this to silence all rgram advisory warnings.


</div>
<!-- END REFERENCE:rgram.RgramWarning -->
<!-- BEGIN REFERENCE:rgram.UnsortedInputWarning -->
<span id="rgram.UnsortedInputWarning"></span>

## rgram.UnsortedInputWarning



<div class="api-class-body" markdown="1">

Input order is valid for estimation but may matter when drawing lines.


</div>
<!-- END REFERENCE:rgram.UnsortedInputWarning -->
<!-- BEGIN REFERENCE:rgram.SupportWarning -->
<span id="rgram.SupportWarning"></span>

## rgram.SupportWarning



<div class="api-class-body" markdown="1">

Predictions or bootstrap intervals have insufficient support.


</div>
<!-- END REFERENCE:rgram.SupportWarning -->
<!-- BEGIN REFERENCE:rgram.NumericalWarning -->
<span id="rgram.NumericalWarning"></span>

## rgram.NumericalWarning



<div class="api-class-body" markdown="1">

A local fit required an explicit numerical fallback.


</div>
<!-- END REFERENCE:rgram.NumericalWarning -->
<!-- BEGIN REFERENCE:rgram.BinningWarning -->
<span id="rgram.BinningWarning"></span>

## rgram.BinningWarning



<div class="api-class-body" markdown="1">

An automatic bin count was bounded to avoid excessive resolution.


</div>
<!-- END REFERENCE:rgram.BinningWarning -->
<!-- BEGIN REFERENCE:rgram.ExtrapolationWarning -->
<span id="rgram.ExtrapolationWarning"></span>

## rgram.ExtrapolationWarning



<div class="api-class-body" markdown="1">

A prediction query lies outside the fitted training range.


</div>
<!-- END REFERENCE:rgram.ExtrapolationWarning -->
<!-- BEGIN REFERENCE:rgram.DataHandlingWarning -->
<span id="rgram.DataHandlingWarning"></span>

## rgram.DataHandlingWarning



<div class="api-class-body" markdown="1">

A numerical rule used a documented fallback or changed resolution.


</div>
<!-- END REFERENCE:rgram.DataHandlingWarning -->

Implementation source: [warnings.py](https://github.com/JackGreenaway/Rgram/blob/main/src/rgram/warnings.py).
