"""
Cost functions cho Linear Regression
"""

from typing import Protocol
import numpy as np


class CostFunction(Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...


class MeanSquaredError:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))


class MeanAbsoluteError:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))


