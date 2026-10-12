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

import numpy as np

from regulo.adam import Adam
from regulo.contracts import TrainingConfig, TrainingResult, ValidationError
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
        config = TrainingConfig(batch, epochs, earlystop, patience)
        self.mlp = mlp
        self.loss = loss
        self.penalty = penalty
        self.adam = adam
        self.batch = config.batch
        self.epochs = config.epochs
        self.earlystop = config.earlystop
        self.patience = config.patience
        if loss.task not in {"regression", "classification"}:
            raise ValidationError(
                "loss.task must be regression or classification."
            )
        self.task = loss.task
        self.history: dict[str, list[float]] = {"train": [], "val": []}
        self._fit_started = False

    def fit(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        xval: np.ndarray | None = None,
        yval: np.ndarray | None = None,
        seed: int | None = None,
    ) -> TrainingResult:
        """Train the network on *xtrain* with optional validation.

        Args:
            xtrain: Training inputs of shape ``(n_train, p)``.
            ytrain: Training targets.
            xval: Validation inputs (optional).  Required for
                early stopping.
            yval: Validation targets (optional).
            seed: Optional integer seed for mini-batch shuffling.
        """
        if self._fit_started:
            raise RuntimeError("Runner.fit() is single-use; call reset() first.")
        if self.earlystop and (xval is None or yval is None):
            raise ValueError("earlystop requires xval and yval to be provided.")
        if (xval is None) != (yval is None):
            raise ValueError("xval and yval must be provided together.")
        if xtrain.ndim != 2 or xtrain.shape[0] == 0:
            raise ValueError("xtrain must be a non-empty 2D array.")
        if ytrain.shape[0] != xtrain.shape[0]:
            raise ValueError(
                "xtrain and ytrain must have the same sample count."
            )
        if xval is not None and yval is not None:
            if xval.ndim != 2 or xval.shape[0] == 0:
                raise ValueError("xval must be a non-empty 2D array.")
            if yval.shape[0] != xval.shape[0]:
                raise ValueError(
                    "xval and yval must have the same sample count."
                )
        if not np.all(np.isfinite(xtrain)) or not np.all(np.isfinite(ytrain)):
            raise ValueError("training data must contain only finite values.")
        if xval is not None and yval is not None:
            if not np.all(np.isfinite(xval)) or not np.all(np.isfinite(yval)):
                raise ValueError(
                    "validation data must contain only finite values."
                )
        self._fit_started = True
        count = xtrain.shape[0]
        rng = np.random.default_rng(seed)
        best = float("inf")
        wait = 0
        best_epoch: int | None = None
        stopped_early = False
        snapshot: (
            tuple[dict[str, list[np.ndarray]], dict[str, object]] | None
        ) = None

        for epoch in range(self.epochs):
            indices = rng.permutation(count)
            losses: list[float] = []
            for start in range(0, count, self.batch):
                end = min(start + self.batch, count)
                chunk = indices[start:end]
                xchunk = xtrain[chunk]
                ychunk = ytrain[chunk]

                ypred = self.mlp(xchunk)
                if not np.all(np.isfinite(ypred)):
                    raise FloatingPointError(
                        f"non-finite prediction at epoch {epoch + 1}, "
                        f"batch {start // self.batch}"
                    )
                cost = self.loss.value(ypred, ychunk)

                # Penalty value and gradient over applicable layers.
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        cost += self.penalty.value(w, i)
                losses.append(cost)
                if not np.isfinite(cost):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch + 1}, "
                        f"batch {start // self.batch}"
                    )

                dloss = self.loss.grad(ypred, ychunk)
                grads = self.mlp.grad(dloss)
                if not all(
                    np.all(np.isfinite(g))
                    for group in grads.values()
                    for g in group
                ):
                    raise FloatingPointError(
                        f"non-finite gradient at epoch {epoch + 1}, "
                        f"batch {start // self.batch}"
                    )
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        grads["weights"][i] = grads["weights"][
                            i
                        ] + self.penalty.grad(w, i)

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
                if not np.all(np.isfinite(valpred)):
                    raise FloatingPointError(
                        f"non-finite validation prediction at epoch {epoch + 1}"
                    )
                valcost = self.loss.value(valpred, yval)
                for i, w in enumerate(self.mlp.weights):
                    if self.penalty.applies(i):
                        valcost += self.penalty.value(w, i)
                self.history["val"].append(float(valcost))

                if self.earlystop:
                    if valcost < best:
                        best = valcost
                        wait = 0
                        best_epoch = epoch
                        snapshot = (self.mlp.state(), self.adam.state())
                    else:
                        wait += 1
                        if wait >= self.patience:
                            if snapshot is not None:
                                self.mlp.load(snapshot[0])
                                self.adam.load_state(snapshot[1])
                            stopped_early = True
                            break

        return TrainingResult(
            tuple(self.history["train"]),
            tuple(self.history["val"]),
            best_epoch,
            stopped_early,
            "patience" if stopped_early else "completed",
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate raw predictions (logits for classification)."""
        return self.mlp(x)

    def proba(self, x: np.ndarray) -> np.ndarray:
        """Generate softmax probabilities (classification only)."""
        if self.task != "classification":
            raise NotImplementedError(
                "proba is only defined for classification (Softmax loss) runners."
            )
        logits = self.mlp(x)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        e = np.exp(shifted)
        return e / np.sum(e, axis=1, keepdims=True)

    def classify(self, x: np.ndarray) -> np.ndarray:
        """Generate class index predictions (classification only)."""
        return np.argmax(self.proba(x), axis=1)

    def reset(self, seed: int | None = None) -> None:
        """Reset the network weights and optimizer state.

        Re-initialises :attr:`mlp` from scratch (with *seed*) and
        clears the optimizer state and training history.
        """
        shape = self.mlp.shape
        self.mlp = MLP(shape, seed=seed)
        self.adam.reset()
        self.history = {"train": [], "val": []}
        self._fit_started = False

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
        with np.load(p / "weights.npz", allow_pickle=False) as weights:
            loaded_weights: list[np.ndarray] = []
            for i, w in enumerate(self.mlp.weights):
                if f"w{i}" not in weights.files:
                    raise ValueError(f"Missing weight array w{i}.")
                loaded = weights[f"w{i}"]
                if loaded.shape != w.shape or not np.all(np.isfinite(loaded)):
                    raise ValueError(f"Invalid weight array w{i}.")
                loaded_weights.append(loaded)
        with np.load(p / "biases.npz", allow_pickle=False) as biases:
            loaded_biases: list[np.ndarray] = []
            for i, b in enumerate(self.mlp.biases):
                if f"b{i}" not in biases.files:
                    raise ValueError(f"Missing bias array b{i}.")
                loaded = biases[f"b{i}"]
                if loaded.shape != b.shape or not np.all(np.isfinite(loaded)):
                    raise ValueError(f"Invalid bias array b{i}.")
                loaded_biases.append(loaded)
        self.mlp.load({"weights": loaded_weights, "biases": loaded_biases})

    def __repr__(self) -> str:
        return (
            f"Runner(mlp={self.mlp!r}, loss={self.loss!r}, "
            f"penalty={self.penalty!r}, adam={self.adam!r})"
        )
