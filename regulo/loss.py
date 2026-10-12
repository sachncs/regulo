"""Loss functions with value and grad methods.

Provides differentiable loss functions used during training.  Each
loss exposes two methods:

* :meth:`Loss.value` -- scalar evaluation
* :meth:`Loss.grad` -- analytical gradient w.r.t. the prediction

The ``1 / n_samples`` averaging convention is applied *inside*
:meth:`Loss.grad` so that gradients can be directly consumed by the
network backward pass.

Gradient scaling convention
---------------------------
Gradients are **summed** across samples in the network backward pass
(:meth:`MLP.grad`) and **divided by n** in the loss backward pass.
This matches standard deep-learning convention: the optimizer
receives ``(1/n) * sum_i nabla L_i``.

Numerical stability
-------------------
:class:`Softmax` clips probabilities at ``1e-15`` before taking the
log, so confident-correct predictions yield exactly ``0.0`` loss
rather than ``log(1 + 1e-15)``.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

__all__ = ["Loss", "Softmax", "Square"]


class Loss(ABC):
    """Abstract base class for differentiable losses.

    Attributes:
        name: Short identifier used by :func:`regulo.tune.resolve`.
    """

    name: ClassVar[str] = ""
    task: ClassVar[str] = ""

    @abstractmethod
    def value(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute the scalar loss."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def grad(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. ``prediction``."""
        raise NotImplementedError  # pragma: no cover

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Square(Loss):
    r"""Mean squared error loss for regression.

    Computes the element-wise mean of squared residuals over all
    scalar entries ``N = prediction.size``:

    .. math::

        L = \frac{1}{N} \sum_{i,j} (y_{ij} - \hat{y}_{ij})^2

    The gradient w.r.t. ``prediction`` is
    ``(2 / N) * (prediction - target)`` -- the same ``N`` appears in
    both :meth:`value` and :meth:`grad`, so the gradient is the
    exact derivative of :meth:`value`.  This matches PyTorch's
    ``MSELoss(reduction='mean')``.
    """

    name: ClassVar[str] = "square"
    task: ClassVar[str] = "regression"

    def value(self, prediction: np.ndarray, target: np.ndarray) -> float:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target must have identical shapes, got "
                f"{prediction.shape} and {target.shape}."
            )
        if prediction.size == 0:
            raise ValueError("prediction and target must be non-empty.")
        if not np.all(np.isfinite(prediction)) or not np.all(
            np.isfinite(target)
        ):
            raise ValueError(
                "prediction and target must contain finite values."
            )
        diff = prediction - target
        return float(np.mean(diff**2))

    def grad(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        if prediction.shape != target.shape or prediction.size == 0:
            raise ValueError(
                "prediction and target must have a non-empty equal shape."
            )
        n = prediction.size
        return 2.0 * (prediction - target) / n


class Softmax(Loss):
    """Softmax cross-entropy loss for multi-class classification.

    Applies a numerically stable softmax internally (subtracting the
    row-max logit before exponentiation) and computes
    ``-log(p[target])`` averaged over samples.  Target labels must
    be integer class indices in ``[0, n_classes)``.
    """

    name: ClassVar[str] = "softmax"
    task: ClassVar[str] = "classification"

    def value(self, logits: np.ndarray, target: np.ndarray) -> float:
        if (
            logits.ndim != 2
            or target.ndim != 1
            or logits.shape[0] != target.size
        ):
            raise ValueError(
                "logits must be 2D and target must match its rows."
            )
        if target.size == 0:
            raise ValueError("target must be non-empty.")
        if target.dtype.kind != "i":
            raise TypeError(
                f"target must be integer dtype, got {target.dtype}."
            )
        if target.min() < 0 or target.max() >= logits.shape[1]:
            raise ValueError(
                f"target values must be in [0, {logits.shape[1]}), "
                f"got [{target.min()}, {target.max()}]."
            )
        # Numerically stable softmax.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(shifted)
        probs = e / np.sum(e, axis=1, keepdims=True)
        # Floor at 1e-15 so confident-correct predictions yield
        # exactly 0.0 loss instead of log(1 + 1e-15).
        clipped = np.clip(probs, 1e-15, 1.0)
        return float(-np.mean(np.log(clipped[np.arange(len(target)), target])))

    def grad(self, logits: np.ndarray, target: np.ndarray) -> np.ndarray:
        if logits.ndim != 2 or target.ndim != 1:
            raise ValueError("logits must be 2D and target must be 1D.")
        if logits.shape[0] != target.shape[0] or target.size == 0:
            raise ValueError(
                "logits and target must have the same non-zero length."
            )
        if target.dtype.kind != "i":
            raise TypeError(
                f"target must be integer dtype, got {target.dtype}."
            )
        if target.min() < 0 or target.max() >= logits.shape[1]:
            raise ValueError("target contains an invalid class index.")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(shifted)
        probs = e / np.sum(e, axis=1, keepdims=True)
        grad = probs.copy()
        grad[np.arange(len(target)), target] -= 1.0
        return grad / logits.shape[0]
