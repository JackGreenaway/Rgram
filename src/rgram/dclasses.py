from typing import Literal
from dataclasses import dataclass


@dataclass
class OlsKws:
    calc_ols: bool = True
    order: int = 1
    add_intercept: bool = True
    ols_y_target: Literal["y_val", "y_pred_rgram", "y_val_cum_sum"] = "y_val"


@dataclass
class CumsumKws:
    calc_cum_sum: bool = True
    reverse: bool = False


@dataclass
class KernelSmoothKws:
    calc_kws: bool = True
    n_eval_points: int = 150
