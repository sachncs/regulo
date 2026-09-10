"""Evaluation metrics for regression and classification.

Each metric is a callable class implementing the :class:`Metric`
abstract interface.  All metrics accept and return scalar floats so
that they can be composed freely in cross-validation loops.

Edge-case conventions
---------------------
* **Empty arrays.**  Functions that compute means (MSE, MAE, RMSE)
  raise ``ZeroDivisionError`` or return ``NaN`` if passed empty
  inputs, consistent with ``numpy.mean``.
* **Perfect predictions (R-squared).**  When ``sstot == 0``
  (constant target), R-squared is ``1.0`` if the residuals are
  also zero, otherwise ``0.0``.
* **No true samples for a class (balanced accuracy).**  Classes
  present only in ``y_pred`` are silently ignored; if ``y_true`` is
  empty the return value is ``0.0``.
"""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Metric", "Mse", "Mae", "Rmse", "R2", "Balanced"]


class Metric(ABC):
    """Abstract base class for scalar evaluation metrics."""

    name: str = ""

    @abstractmethod
    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        """Compute the metric value."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Mse(Metric):
    """Mean squared error: ``(1/n) * sum((truth - pred)**2)``.

    Computed over the total scalar element count.  Compatible with
    ``numpy.mean`` semantics for arbitrary-shape arrays.
    """

    name = "mse"

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        return float(np.mean((truth - pred) ** 2))


class Mae(Metric):
    """Mean absolute error: ``(1/n) * sum(|truth - pred|)``."""

    name = "mae"

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        return float(np.mean(np.abs(truth - pred)))


class Rmse(Metric):
    """Root mean squared error: ``sqrt(MSE)``.

    In the same units as the target.  Implemented via composition
    with :class:`Mse` rather than a separate formula so there is
    exactly one source of truth for the MSE computation.
    """

    name = "rmse"

    def __init__(self) -> None:
        self.mse = Mse()

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        return float(np.sqrt(self.mse(truth, pred)))


class R2(Metric):
    """Coefficient of determination.

    ``R^2 = 1 - SS_res / SS_tot`` where
    ``SS_res = sum((y - yhat)^2)`` and
    ``SS_tot = sum((y - mean(y))^2)``.

    Edge cases:
    * If ``SS_tot == 0`` and ``SS_res == 0``, returns ``1.0``.
    * If ``SS_tot == 0`` and ``SS_res > 0``, returns ``0.0``.
    """

    name = "r2"

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        ssres = float(np.sum((truth - pred) ** 2))
        sstot = float(np.sum((truth - np.mean(truth)) ** 2))
        if sstot == 0.0:
            return 1.0 if ssres == 0.0 else 0.0
        return 1.0 - ssres / sstot


class Balanced(Metric):
    """Balanced accuracy for multi-class classification.

    Unweighted mean of per-class recall (sensitivity).  Classes
    present only in ``pred`` are silently ignored.  Empty inputs
    return ``0.0``.
    """

    name = "balanced"

    def __call__(self, truth: np.ndarray, pred: np.ndarray) -> float:
        classes = np.unique(truth)
        recall = []
        for c in classes:
            mask = truth == c
            if np.any(mask):
                recall.append(float(np.mean(pred[mask] == c)))
        return float(np.mean(recall)) if recall else 0.0
