"""Manual feedforward ReLU network with back-propagation.

Implements a fully connected multi-layer perceptron using only
NumPy.  All forward and backward passes are explicit -- there is no
autograd engine.  Weight initialization follows the Xavier (Glorot)
uniform scheme.

Architecture
------------
The network is defined by a list of layer widths
``[input, hidden_1, ..., output]``.  Hidden layers use ReLU
activations; the output layer is linear (no activation).  This is
the standard architecture for regression; for classification the
output is passed through a softmax in the loss function (see
:class:`regulo.loss.Softmax`).

Back-propagation
----------------
The backward pass computes exact gradients w.r.t. all weights and
biases using the chain rule.  It requires cached activations from
the forward pass (stored in :attr:`MLP.pre` and :attr:`MLP.post`),
so :meth:`MLP.__call__` must be invoked before :meth:`MLP.grad`.
"""

import numpy as np

__all__ = ["MLP", "xavier"]


def xavier(fanin: int, fanout: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a weight matrix from the Xavier uniform distribution.

    Draws each entry independently from ``Uniform(-limit, limit)``
    with ``limit = sqrt(6 / (fanin + fanout))``.  The caller
    supplies a NumPy :class:`~numpy.random.Generator` to make
    initialization reproducible across runs.

    Args:
        fanin: Number of input features (``W.shape[0]``).
        fanout: Number of output features (``W.shape[1]``).
        rng: NumPy random generator used for sampling.

    Returns:
        Weight matrix of shape ``(fanin, fanout)``.
    """
    limit = np.sqrt(6.0 / (fanin + fanout))
    return rng.uniform(-limit, limit, size=(fanin, fanout))


class MLP:
    """Feedforward MLP with ReLU hidden activations and a linear output.

    The network stores its own forward-pass activations in semi-private
    caches so that :meth:`grad` can compute exact gradients without
    re-running the forward pass. A single MLP instance is not re-entrant;
    use one instance per concurrent training operation.

    Attributes:
        shape: List of layer widths ``[input, hidden_1, ..., output]``.
        layers: Number of weight matrices (``len(shape) - 1``).
        weights: List of weight matrices, one per layer, in
            input-to-output order.
        biases: List of row-vector biases, one per layer.
    """

    def __init__(self, shape: list[int], seed: int | None = None) -> None:
        """Initialise weights with Xavier-uniform and biases to zero.

        Args:
            shape: Layer widths including input and output.
            seed: Optional integer seed for the NumPy random
                generator used by :func:`xavier`.  ``None`` defers to
                NumPy's default seeding.
        """
        if len(shape) < 2:
            raise ValueError("shape must have at least 2 elements.")
        if any(s <= 0 for s in shape):
            raise ValueError("All layer widths must be positive.")
        self.shape = list(shape)
        self.layers = len(shape) - 1
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(self.layers):
            self.weights.append(xavier(shape[i], shape[i + 1], rng))
            self.biases.append(np.zeros((1, shape[i + 1])))

        self._pre: list[np.ndarray] = []
        self._post: list[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass, caching intermediates for :meth:`grad`."""
        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")
        if x.shape[1] != self.shape[0]:
            raise ValueError(
                f"x has {x.shape[1]} features; expected {self.shape[0]}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values.")
        self._pre = []
        self._post = [x]
        a = x
        for i in range(self.layers):
            z = a @ self.weights[i] + self.biases[i]
            self._pre.append(z)
            if i < self.layers - 1:
                a = np.maximum(z, 0.0)
            else:
                a = z
            self._post.append(a)
        return a

    def grad(self, dloss: np.ndarray) -> dict[str, list[np.ndarray]]:
        """Compute parameter gradients via back-propagation.

        Assumes :meth:`__call__` has been invoked immediately before
        and that ``dloss`` already contains the ``1 / n_samples``
        scaling from the loss backward pass.  The returned
        gradients are **not** scaled by the learning rate -- that
        is the optimizer's job.

        Returns:
            Dictionary with ``"weights"`` and ``"biases"`` lists,
            each in input-to-output order.
        """
        if not self._post:
            raise RuntimeError("__call__ must be invoked before grad().")
        wgrads: list[np.ndarray] = []
        bgrads: list[np.ndarray] = []
        delta = dloss
        for i in reversed(range(self.layers)):
            prev = self._post[i]
            dw = prev.T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            wgrads.insert(0, dw)
            bgrads.insert(0, db)
            if i > 0:
                delta = delta @ self.weights[i].T
                z = self._pre[i - 1]
                delta *= (z > 0.0).astype(delta.dtype)
        return {"weights": wgrads, "biases": bgrads}

    def state(self) -> dict[str, list[np.ndarray]]:
        """Return deep copies of all current parameters."""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def load(self, params: dict[str, list[np.ndarray]]) -> None:
        """Replace all parameters (deep-copied from *params*)."""
        if set(params) != {"weights", "biases"}:
            raise ValueError("params must contain weights and biases groups.")
        if (
            len(params["weights"]) != self.layers
            or len(params["biases"]) != self.layers
        ):
            raise ValueError("params must contain one array per network layer.")
        for i, (weight, bias) in enumerate(
            zip(params["weights"], params["biases"], strict=False)
        ):
            if weight.shape != (self.shape[i], self.shape[i + 1]):
                raise ValueError(f"Invalid weight shape for layer {i}.")
            if bias.shape != (1, self.shape[i + 1]):
                raise ValueError(f"Invalid bias shape for layer {i}.")
            if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
                raise ValueError("Network parameters must be finite.")
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]

    def __repr__(self) -> str:
        return f"MLP(shape={self.shape})"
