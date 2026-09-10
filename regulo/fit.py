"""Training loop with mini-batching, early stopping, and validation.

The :class:`Runner` class orchestrates the epoch-level training
loop:

* Mini-batch sampling with per-epoch shuffling.
* Forward pass, loss computation, and penalty value.
* Back-propagation and parameter update via :class:`regulo.adam.Adam`.
* Optional validation loss tracking and early stopping.

Covridge / Sparridge layer handling
------------------------------------
:class:`Covridge` and :class:`Sparridge` are applied only to the
first weight matrix because the empirical Gram matrix
``C_{delta,n}`` is defined over the input dimension and therefore
only matches ``W^{(1)}``.  The dispatch uses the polymorphic
:meth:`Penalty.applies` hook -- no ``isinstance`` checks.

Gradient flow
-------------
Each update step computes:

1. ``y_pred = mlp(xchunk)``
2. ``cost = loss.value(ypred, ychunk)``
3. ``dloss = loss.grad(ypred, ychunk)``
4. ``grads = mlp.grad(dloss)``
5. ``grads["weights"][i] += penalty.grad(weight_i, i)`` for layers
   where ``penalty.applies(i)``.
6. ``new_params = adam.step(params, grads)``

Lifecycle
---------
A :class:`Runner` instance is single-use by design: it owns the
network weights, optimizer state, and history.  To re-train from
scratch, call :meth:`reset` or construct a fresh :class:`Runner`.
"""

from typing import Dict, List, Optional

import numpy as np

from regulo.adam import Adam
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import Penalty

__all__ = ["Runner"]


class Runner:
    """End-to-end trainer for the manual NumPy network.

    Ties together a :class:`MLP`, a :class:`Loss`, a :class:`Penalty`,
    and an :class:`Adam` optimizer into a single
    :meth:`fit` / :meth:`predict` interface.

    Attributes:
        task: ``"regression"`` or ``"classification"``, derived from
            the loss type.
        history: Dictionary with ``"train"`` and ``"val"`` lists
            populated during :meth:`fit`.
    """

    def __init__(
        self,
        mlp: MLP,
        loss: Loss,
        penalty: Penalty,
        adam: Adam,
        batch: int = 32,
        epochs: int = 500,
        earlystop: bool = False,
        patience: int = 10,
    ) -> None:
        """Initialise the runner with all component objects."""
        if batch <= 0:
            raise ValueError("batch must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if patience <= 0:
            raise ValueError("patience must be positive.")
        self.mlp = mlp
        self.loss = loss
        self.penalty = penalty
        self.adam = adam
        self.batch = batch
        self.epochs = epochs
        self.earlystop = earlystop
        self.patience = patience
        self.task = "classification" if loss.name == "softmax" else "regression"
        self.history: Dict[str, List[float]] = {"train": [], "val": []}

    def fit(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        xval: Optional[np.ndarray] = None,
        yval: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Train the network on *xtrain* with optional validation.

        Args:
            xtrain: Training inputs of shape ``(n_train, p)``.
            ytrain: Training targets.
            xval: Validation inputs (optional).  Required for
                early stopping.
            yval: Validation targets (optional).
            seed: Optional integer seed for mini-batch shuffling.
        """
        if self.earlystop and (xval is None or yval is None):
            raise ValueError(
                "earlystop requires xval and yval to be provided."
            )
        count = xtrain.shape[0]
        rng = np.random.default_rng(seed)
        best = float("inf")
        wait = self.patience
        snapshot: Optional[Dict[str, List[np.ndarray]]] = None

        for epoch in range(self.epochs):
            indices = rng.permutation(count)
            losses: List[float] = []
            for start in range(0, count, self.batch):
                end = min(start + self.batch, count)
                chunk = indices[start:end]
                xchunk = xtrain[chunk]
                ychunk = ytrain[chunk]

                ypred = self.mlp(xchunk)
                cost = self.loss.value(ypred, ychunk)

                # Penalty value and gradient over applicable layers.
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        cost += self.penalty.value(w, i)
                losses.append(cost)

                dloss = self.loss.grad(ypred, ychunk)
                grads = self.mlp.grad(dloss)
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        grads["weights"][i] = grads["weights"][i] + self.penalty.grad(
                            w, i
                        )

                params = {
                    "weights": self.mlp.weights,
                    "biases": self.mlp.biases,
                }
                updated = self.adam.step(params, grads)
                # NaN / Inf guard: detect non-finite parameters or
                # gradients and abort with a clear error pointing to
                # the offending epoch and batch index.
                for i, w in enumerate(updated["weights"]):
                    if not np.all(np.isfinite(w)):
                        raise FloatingPointError(
                            f"non-finite weight at epoch {epoch + 1}, "
                            f"batch {start // self.batch}, layer {i}"
                        )
                for i, b in enumerate(updated["biases"]):
                    if not np.all(np.isfinite(b)):
                        raise FloatingPointError(
                            f"non-finite bias at epoch {epoch + 1}, "
                            f"batch {start // self.batch}, layer {i}"
                        )
                self.mlp.load(updated)

            self.history["train"].append(float(np.mean(losses)))

            if xval is not None and yval is not None:
                valpred = self.mlp(xval)
                valcost = self.loss.value(valpred, yval)
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        valcost += self.penalty.value(w, i)
                self.history["val"].append(float(valcost))

                if self.earlystop:
                    if valcost < best:
                        best = valcost
                        wait = self.patience
                        snapshot = self.mlp.state()
                    else:
                        wait -= 1
                        if wait < 0:
                            if snapshot is not None:
                                self.mlp.load(snapshot)
                            break

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate raw predictions (logits for classification)."""
        return self.mlp(x)

    def proba(self, x: np.ndarray) -> np.ndarray:
        """Generate softmax probabilities (classification only)."""
        if self.task != "classification":
            raise NotImplementedError(
                "proba is only defined for classification "
                "(Softmax loss) runners."
            )
        logits = self.mlp(x)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / np.sum(e, axis=1, keepdims=True)

    def classify(self, x: np.ndarray) -> np.ndarray:
        """Generate class index predictions (classification only)."""
        return np.argmax(self.proba(x), axis=1)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the network weights and optimizer state.

        Re-initialises :attr:`mlp` from scratch (with *seed*) and
        clears the optimizer state and training history.
        """
        shape = self.mlp.shape
        self.mlp = MLP(shape, seed=seed)
        self.adam.reset()
        self.history = {"train": [], "val": []}

    def restart(self, path: str) -> None:
        """Load weights only from a directory produced by :func:`save`.

        Validates the architecture recorded in ``meta.json`` against
        :attr:`mlp.shape` and replaces the current weights
        without disturbing optimizer state.
        """
        import json
        from pathlib import Path

        p = Path(path)
        with open(p / "meta.json") as f:
            data = json.load(f)
        if list(data["shape"]) != list(self.mlp.shape):
            raise ValueError(
                f"Architecture mismatch: on-disk {data['shape']} "
                f"!= in-memory {self.mlp.shape}."
            )
        weights = np.load(p / "weights.npz")
        biases = np.load(p / "biases.npz")
        for i, w in enumerate(self.mlp.weights):
            w[...] = weights[f"w{i}"]
        for i, b in enumerate(self.mlp.biases):
            b[...] = biases[f"b{i}"]

    def __repr__(self) -> str:
        return (
            f"Runner(mlp={self.mlp!r}, loss={self.loss!r}, "
            f"penalty={self.penalty!r}, adam={self.adam!r})"
        )
