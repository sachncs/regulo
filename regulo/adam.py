"""Adam optimizer implemented in pure NumPy.

Implements the Adam algorithm (Kingma & Ba, 2014) with per-parameter
first- and second-moment estimates, bias-corrected step sizes, and
an explicit :meth:`reset` for re-training from scratch.

The optimizer operates on dictionaries of parameter lists (matching
the format returned by :meth:`regulo.net.MLP.state`) so it is
agnostic to network architecture.

References
----------
Kingma, D. P. & Ba, J. (2014).  "Adam: A Method for Stochastic
Optimization."  *ICLR 2015*.  arXiv:1412.6980.
"""

from typing import Dict, List, Optional

import numpy as np

__all__ = ["Adam"]


class Adam:
    """Adam optimizer with adaptive learning rates.

    Maintains exponential moving averages of past gradients (first
    moment, :attr:`mean`) and past squared gradients (second
    moment, :attr:`variance`).  At each step the bias-corrected
    estimates are used.

    Attributes:
        lr: Step size ``eta``.
        beta1: Exponential decay rate for the first moment.
        beta2: Exponential decay rate for the second moment.
        epsilon: Small constant added to the denominator for
            numerical stability.
        clock: Internal step counter (incremented on each
            :meth:`step` call).
        mean: Per-parameter first-moment buffers (lazily created).
        variance: Per-parameter second-moment buffers (lazily
            created).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialise Adam hyper-parameters and empty moment buffers."""
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("beta1 must be in [0, 1).")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("beta2 must be in [0, 1).")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if lr <= 0.0:
            raise ValueError("lr must be positive.")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clock = 0
        self.mean: Dict[str, List[Optional[np.ndarray]]] = {}
        self.variance: Dict[str, List[Optional[np.ndarray]]] = {}

    def step(
        self,
        params: Dict[str, List[np.ndarray]],
        grads: Dict[str, List[np.ndarray]],
    ) -> Dict[str, List[np.ndarray]]:
        """Perform one Adam update and return the new parameters.

        On first contact with each parameter array the corresponding
        :attr:`mean` and :attr:`variance` buffers are lazily created
        as zero arrays of identical shape.

        The step counter :attr:`clock` is incremented *before* bias
        correction, so that ``clock = 1`` on the very first call.
        The first step therefore uses
        ``mean_hat = mean / (1 - beta1)`` and
        ``variance_hat = variance / (1 - beta2)``, which is the
        correct bias-correction formula in the paper.

        Returns:
            A new parameter dictionary with the same structure as
            ``params``, containing the post-update arrays.
        """
        self.clock += 1
        next: Dict[str, List[np.ndarray]] = {}
        for key in params:
            next[key] = []
            if key not in self.mean:
                self.mean[key] = []
                self.variance[key] = []
            while len(self.mean[key]) < len(params[key]):
                self.mean[key].append(None)
                self.variance[key].append(None)
            for i, (p, g) in enumerate(zip(params[key], grads[key])):
                if self.mean[key][i] is None:
                    self.mean[key][i] = np.zeros_like(p)
                    self.variance[key][i] = np.zeros_like(p)
                m = self.mean[key][i]
                v = self.variance[key][i]
                assert m is not None and v is not None
                m = self.beta1 * m + (1.0 - self.beta1) * g
                v = self.beta2 * v + (1.0 - self.beta2) * (g**2)
                self.mean[key][i] = m
                self.variance[key][i] = v
                mhat = m / (1.0 - self.beta1**self.clock)
                vhat = v / (1.0 - self.beta2**self.clock)
                updated = p - self.lr * mhat / (
                    np.sqrt(vhat) + self.epsilon
                )
                next[key].append(updated)
        return next

    def reset(self) -> None:
        """Reset all internal state to the initial condition."""
        self.clock = 0
        self.mean = {}
        self.variance = {}

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, "
            f"beta1={self.beta1}, beta2={self.beta2}, "
            f"epsilon={self.epsilon})"
        )
