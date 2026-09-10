"""Regularizer penalties and analytical gradients.

This module defines the regularizers used throughout the regulo
library.  Each regularizer is a stateless callable object exposing
two methods:

* :meth:`Penalty.value` -- scalar penalty ``Omega(W)`` added to the
  empirical loss.
* :meth:`Penalty.grad` -- analytical gradient ``nabla_W Omega(W)``
  of the penalty with respect to the weight matrix.

Subclasses may additionally override :meth:`Penalty.applies` to
restrict the penalty to specific weight matrices (used by the
geometry-aware :class:`Covridge` and :class:`Sparridge`).

The concrete penalties map directly to equations in Qasim & Javed:

==========================  ============================================
Class                       Paper equation
==========================  ============================================
:class:`Ridge`              ``lambda ||W||_F^2``
:class:`Lasso`              ``gamma ||W||_1``
:class:`ElasticNet`         ``alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2``
:class:`Covridge`           ``lambda1 ||C^{1/2} W||_F^2 + lambda2 ||W||_F^2``
:class:`Sparridge`          ``lambda1 ||C^{1/2} W||_F^2 + gamma ||W||_1``
==========================  ============================================

Numerical stability
-------------------
:class:`Covridge` and :class:`Sparridge` factor ``C_{delta,n}`` once
at construction using a symmetric eigendecomposition
(``numpy.linalg.eigh``).  The decomposition is symmetric by
construction (``C_n`` and ``I`` are symmetric and so is their sum),
which is why ``eigh`` is preferred over ``scipy.linalg.sqrtm``:
``eigh`` returns real eigenvalues and avoids the complex round-trip
that ``sqrtm`` performs.  Tiny negative eigenvalues introduced by
floating-point error are clamped to zero before the square root is
taken, guaranteeing a real PSD square root.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

__all__ = [
    "Penalty",
    "Void",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Covridge",
    "Sparridge",
    "REGISTRY",
]


class Penalty(ABC):
    """Abstract base class for weight-matrix penalties.

    All penalties share the same interface so that :class:`Runner`
    can dispatch them uniformly.  Subclasses must implement
    :meth:`value` and :meth:`grad`; both are pure functions of the
    weight matrix and have no side effects.

    Geometry-aware penalties (those that depend on a fixed input
    Gram matrix ``C``) override :meth:`applies` to limit their
    scope to the first weight matrix only.

    Attributes:
        name: Short identifier used by :func:`regulo.tune.resolve`.
        hp: Tuple of constructor keyword names accepted by the
            penalty's ``__init__``.
    """

    name: ClassVar[str] = ""
    hp: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def value(self, weight: np.ndarray, layer: int) -> float:
        """Compute the scalar penalty.

        Args:
            weight: Weight matrix of shape ``(in_features, out_features)``.
            layer: Index of the layer the weight belongs to.  Penalties
                that ignore the layer index (the default) accept this
                argument for interface uniformity.

        Returns:
            Scalar penalty value ``Omega(W)`` to be added to the
            empirical loss.
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, weight: np.ndarray, layer: int) -> np.ndarray:
        """Compute the gradient of the penalty w.r.t. ``weight``.

        Args:
            weight: Weight matrix of shape ``(in_features, out_features)``.
            layer: Index of the layer the weight belongs to.

        Returns:
            Gradient array with the same shape as ``weight``.
        """
        raise NotImplementedError

    def applies(self, layer: int) -> bool:
        """Return whether the penalty applies to *layer*.

        Default implementation returns ``True`` for every layer.
        Geometry-aware penalties override this to restrict themselves
        to the first layer (``layer == 0``) where the empirical
        Gram matrix ``C_{delta,n}`` is defined.
        """
        return True

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={getattr(self, k)!r}" for k in self.hp if hasattr(self, k)
        )
        return f"{type(self).__name__}({params})"


class Void(Penalty):
    """No-op penalty used as the unregularized baseline.

    Always returns zero value and zero gradient so that
    :class:`Runner` can iterate over penalty instances without
    special-casing the absence of regularization.
    """

    name: ClassVar[str] = "none"
    hp: ClassVar[tuple[str, ...]] = ()

    def value(self, weight: np.ndarray, layer: int) -> float:
        return 0.0

    def grad(self, weight: np.ndarray, layer: int) -> np.ndarray:
        return np.zeros_like(weight)


class Ridge(Penalty):
    r"""Ridge (L2 / Tikhonov) penalty: ``lambda ||W||_F^2``.

    Equivalent to a Gaussian prior on the weights.  Smooth and
    strongly convex in ``W``; encourages small magnitudes without
    inducing sparsity.  Setting ``lam = 0`` reduces this to
    :class:`Void`.
    """

    name: ClassVar[str] = "ridge"
    hp: ClassVar[tuple[str, ...]] = ("lam",)

    def __init__(self, lam: float) -> None:
        """Store the non-negative regularization strength."""
        if lam < 0:
            raise ValueError("lam must be non-negative.")
        self.lam = lam

    def value(self, weight: np.ndarray, layer: int) -> float:
        return self.lam * float(np.sum(weight**2))

    def grad(self, weight: np.ndarray, layer: int) -> np.ndarray:
        return 2.0 * self.lam * weight


class Lasso(Penalty):
    r"""Lasso (L1) penalty: ``gamma ||W||_1``.

    Promotes sparsity by driving individual weights exactly to
    zero.  Non-smooth at the origin, so the implementation uses a
    subgradient rather than a true gradient.  ``numpy.sign(0) == 0``
    so the subgradient is well-defined at ``W = 0``.
    """

    name: ClassVar[str] = "lasso"
    hp: ClassVar[tuple[str, ...]] = ("gamma",)

    def __init__(self, gamma: float) -> None:
        """Store the non-negative sparsity weight."""
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = gamma

    def value(self, weight: np.ndarray, layer: int) -> float:
        return self.gamma * float(np.sum(np.abs(weight)))

    def grad(self, weight: np.ndarray, layer: int) -> np.ndarray:
        return self.gamma * np.sign(weight)


class ElasticNet(Penalty):
    r"""Elastic Net: ``alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2``.

    Convex combination of :class:`Lasso` (sparsity) and :class:`Ridge`
    (shrinkage).  At ``alpha = 0`` it degenerates to a pure quadratic
    penalty; at ``alpha = 1`` it degenerates to pure L1.
    """

    name: ClassVar[str] = "elastic_net"
    hp: ClassVar[tuple[str, ...]] = ("alpha", "gamma")

    def __init__(self, alpha: float, gamma: float) -> None:
        """Store mixing parameter and penalty weight."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma

    def value(self, weight: np.ndarray, layer: int) -> float:
        l1 = self.alpha * self.gamma * float(np.sum(np.abs(weight)))
        l2 = (1.0 - self.alpha) * 0.5 * float(np.sum(weight**2))
        return l1 + l2

    def grad(self, weight: np.ndarray, layer: int) -> np.ndarray:
        l1 = self.alpha * self.gamma * np.sign(weight)
        l2 = (1.0 - self.alpha) * weight
        return l1 + l2


class Covridge(Penalty):
    r"""Covridge penalty: ``lambda1 ||C^{1/2} W||_F^2 + lambda2 ||W||_F^2``.

    Geometry-aware shrinkage along the eigenvectors of the empirical
    Gram matrix ``C_n = (1/n) X^T X`` stabilized by ``delta I_p`` to
    form ``C_{delta,n}``.  The second term is a standard isotropic
    ridge.  Applies only to the first weight matrix because
    ``C_{delta,n}`` is defined over the input dimension.
    """

    name: ClassVar[str] = "covridge"
    hp: ClassVar[tuple[str, ...]] = ("lambda1", "lambda2", "gram")

    def __init__(
        self,
        lambda1: float,
        lambda2: float,
        gram: np.ndarray,
    ) -> None:
        """Store penalties and precompute ``C_{delta,n}^{1/2}``."""
        if lambda1 < 0 or lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be non-negative.")
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # Symmetric eigendecomposition: eigh returns real eigenvalues
        # and avoids the complex round-trip of sqrtm.
        eigvals, eigvecs = np.linalg.eigh(gram)
        eigvals = np.maximum(eigvals, 0.0)
        self.csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def applies(self, layer: int) -> bool:
        return layer == 0

    def value(self, weight: np.ndarray, layer: int) -> float:
        cw = self.csqrt @ weight
        return self.lambda1 * float(np.sum(cw**2)) + self.lambda2 * float(
            np.sum(weight**2)
        )

    def grad(self, weight: np.ndarray, layer: int) -> np.ndarray:
        c = self.csqrt @ self.csqrt
        return 2.0 * self.lambda1 * (c @ weight) + 2.0 * self.lambda2 * weight


class Sparridge(Penalty):
    r"""Sparridge penalty: ``lambda1 ||C^{1/2} W||_F^2 + gamma ||W||_1``.

    Combines the geometry-aware shrinkage of :class:`Covridge` with
    the sparsity-inducing L1 term of :class:`Lasso`.  Applies only
    to the first weight matrix.
    """

    name: ClassVar[str] = "sparridge"
    hp: ClassVar[tuple[str, ...]] = ("lambda1", "gamma", "gram")

    def __init__(
        self,
        lambda1: float,
        gamma: float,
        gram: np.ndarray,
    ) -> None:
        """Store penalties and precompute ``C_{delta,n}^{1/2}``."""
        if lambda1 < 0 or gamma < 0:
            raise ValueError("lambda1 and gamma must be non-negative.")
        self.lambda1 = lambda1
        self.gamma = gamma
        eigvals, eigvecs = np.linalg.eigh(gram)
        eigvals = np.maximum(eigvals, 0.0)
        self.csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def applies(self, layer: int) -> bool:
        return layer == 0

    def value(self, weight: np.ndarray, layer: int) -> float:
        cw = self.csqrt @ weight
        return self.lambda1 * float(np.sum(cw**2)) + self.gamma * float(
            np.sum(np.abs(weight))
        )

    def grad(self, weight: np.ndarray, layer: int) -> np.ndarray:
        c = self.csqrt @ self.csqrt
        l2 = 2.0 * self.lambda1 * (c @ weight)
        l1 = self.gamma * np.sign(weight)
        return l2 + l1


REGISTRY: dict[str, type[Penalty]] = {
    cls.name: cls
    for cls in (Void, Ridge, Lasso, ElasticNet, Covridge, Sparridge)
}
