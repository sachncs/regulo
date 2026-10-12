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
        if not np.isfinite(beta1) or not (0.0 <= beta1 < 1.0):
            raise ValueError("beta1 must be in [0, 1).")
        if not np.isfinite(beta2) or not (0.0 <= beta2 < 1.0):
            raise ValueError("beta2 must be in [0, 1).")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be positive.")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clock = 0
        self.mean: dict[str, list[np.ndarray | None]] = {}
        self.variance: dict[str, list[np.ndarray | None]] = {}

    def step(
        self,
        params: dict[str, list[np.ndarray]],
        grads: dict[str, list[np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
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
        if set(params) != set(grads):
            raise ValueError("params and grads must have identical groups.")
        for key, values in params.items():
            if len(values) != len(grads[key]):
                raise ValueError(
                    f"Mismatched parameter count for group {key!r}."
                )
            for param, grad in zip(values, grads[key], strict=False):
                if param.shape != grad.shape:
                    raise ValueError(
                        f"Mismatched parameter shape for group {key!r}."
                    )
                if not np.all(np.isfinite(grad)):
                    raise FloatingPointError(
                        f"Non-finite gradient in group {key!r}."
                    )

        self.clock += 1
        updated_params: dict[str, list[np.ndarray]] = {}
        for key, values in params.items():
            updated_params[key] = []
            if key not in self.mean:
                self.mean[key] = []
                self.variance[key] = []
            while len(self.mean[key]) < len(values):
                self.mean[key].append(None)
                self.variance[key].append(None)
            for i, (p, g) in enumerate(zip(values, grads[key], strict=False)):
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
                updated = p - self.lr * mhat / (np.sqrt(vhat) + self.epsilon)
                updated_params[key].append(updated)
        return updated_params

    def reset(self) -> None:
        """Reset all internal state to the initial condition."""
        self.clock = 0
        self.mean = {}
        self.variance = {}

    def state(self) -> dict[str, object]:
        """Return a deep-copyable snapshot of optimizer state."""
        return {
            "clock": self.clock,
            "mean": {
                key: [
                    None if value is None else value.copy() for value in values
                ]
                for key, values in self.mean.items()
            },
            "variance": {
                key: [
                    None if value is None else value.copy() for value in values
                ]
                for key, values in self.variance.items()
            },
        }

    def load_state(self, state: dict[str, object]) -> None:
        """Restore a snapshot created by :meth:`state`."""
        clock = state.get("clock")
        mean = state.get("mean")
        variance = state.get("variance")
        if (
            not isinstance(clock, int)
            or clock < 0
            or not isinstance(mean, dict)
            or not isinstance(variance, dict)
            or set(mean) != set(variance)
        ):
            raise ValueError("Invalid Adam state snapshot.")
        for group in mean:
            moments = mean[group]
            squared = variance[group]
            if not isinstance(moments, list) or not isinstance(squared, list):
                raise ValueError("Invalid Adam state buffer list.")
            if len(moments) != len(squared):
                raise ValueError(
                    "Adam moment buffers have inconsistent lengths."
                )
            for first, second in zip(moments, squared, strict=False):
                if first is None or second is None:
                    continue
                if not isinstance(first, np.ndarray) or not isinstance(
                    second, np.ndarray
                ):
                    raise ValueError(
                        "Adam buffers must be NumPy arrays or None."
                    )
                if (
                    first.shape != second.shape
                    or not np.all(np.isfinite(first))
                    or not np.all(np.isfinite(second))
                ):
                    raise ValueError(
                        "Adam buffers must have equal finite shapes."
                    )
        self.clock = clock
        self.mean = mean  # type: ignore[assignment]
        self.variance = variance  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, "
            f"beta1={self.beta1}, beta2={self.beta2}, "
            f"epsilon={self.epsilon})"
        )
