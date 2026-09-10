"""Cross-validation and hyperparameter grid search utilities.

Provides primitives for k-fold splitting, feature standardization,
penalty construction from method-name dispatch, and exhaustive
grid search over hyperparameter dictionaries.
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np

from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import Penalty, REGISTRY

__all__ = ["kfold", "Scaler", "resolve", "search"]


def kfold(
    n: int, folds: int, seed: Optional[int] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train, val)`` index tuples for *folds* folds.

    Each fold uses a contiguous slice of a shuffled permutation;
    shuffling is seeded by *seed* for reproducibility.  Standard
    K-fold behaviour with shuffle.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if folds > n:
        raise ValueError("folds cannot exceed n.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    sizes = [n // folds] * folds
    for i in range(n % folds):
        sizes[i] += 1
    start = 0
    for size in sizes:
        val = perm[start : start + size]
        train = np.concatenate([perm[:start], perm[start + size :]])
        yield train, val
        start += size


class Scaler:
    """Per-column z-score scaler fit on one array, applied to others.

    Computes and stores the mean and standard deviation of each
    column on :meth:`fit`, then applies the transformation via
    :meth:`transform`.  :meth:`fittransform` combines the two for
    convenience.
    """

    name = "scaler"

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "Scaler":
        """Compute per-column mean and standard deviation."""
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        # Guard against constant columns (std == 0) so transform does
        # not produce NaN.
        if self.std is not None:
            self.std = np.where(self.std == 0.0, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fit.")
        return (x - self.mean) / self.std

    def fittransform(self, x: np.ndarray) -> np.ndarray:
        """Fit on *x* and return the transformed array."""
        self.fit(x)
        return self.transform(x)

    def __repr__(self) -> str:
        if self.mean is None:
            return "Scaler(unfit)"
        return f"Scaler(mean={self.mean.tolist()!r}, std={self.std.tolist()!r})"


def resolve(
    name: str,
    hp: dict,
    xtrain: np.ndarray,
    delta: float = 1e-4,
) -> Penalty:
    """Construct a :class:`Penalty` from a method name and hyperparameters.

    For geometry-aware methods (``"covridge"``, ``"sparridge"``)
    the stabilized Gram matrix ``C_{delta,n}`` is computed from
    *xtrain* and passed to the constructor.  The Gram matrix is
    also computed unconditionally for non-geometry methods; the
    small overhead is the price of dispatch simplicity.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown method: {name!r}. Known: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    unknown = set(hp) - set(cls.hp)
    if unknown:
        raise ValueError(
            f"Method {name!r} accepts keys {cls.hp}, "
            f"but got unexpected {sorted(unknown)}."
        )
    n, p = xtrain.shape
    gram = (xtrain.T @ xtrain) / n + delta * np.eye(p)
    hp = dict(hp)
    if "gram" in cls.hp:
        hp.setdefault("gram", gram)
    return cls(**hp)


def search(
    x: np.ndarray,
    y: np.ndarray,
    shape: List[int],
    method: str,
    grid: List[dict],
    loss: Loss,
    folds: int = 5,
    batch: int = 32,
    epochs: int = 500,
    lr: float = 1e-3,
    earlystop: bool = False,
    patience: int = 10,
    seed: Optional[int] = None,
) -> Tuple[Optional[dict], float]:
    """Run k-fold cross-validation over a hyperparameter grid.

    Each combination in *grid* is evaluated on *folds*
    folds.  A fresh network, penalty, and optimizer are created
    per fold to avoid state leakage.

    The scoring metric is derived from ``loss``: negative MSE for
    :class:`regulo.loss.Square`, balanced accuracy for
    :class:`regulo.loss.Softmax`.

    Returns:
        ``(best, score)`` -- the hyperparameters that
        achieved the highest mean fold score and that score.
    """
    if not grid:
        raise ValueError("grid must contain at least one entry.")
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    task = "classification" if loss.name == "softmax" else "regression"
    top = -float("inf")
    best: Optional[dict] = None
    from regulo.score import Balanced, Mse

    metric = Balanced() if task == "classification" else Mse()

    for params in grid:
        scores: List[float] = []
        for train, val in kfold(x.shape[0], folds, seed=seed):
            xtrain = x[train]
            xval = x[val]
            ytrain = y[train]
            yval = y[val]

            scaler = Scaler().fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xval = scaler.transform(xval)

            penalty = resolve(method, params, xtrain)
            mlp = MLP(shape, seed=seed)
            adam = Adam(lr=lr)
            runner = Runner(
                mlp,
                loss,
                penalty,
                adam,
                batch=batch,
                epochs=epochs,
                earlystop=earlystop,
                patience=patience,
            )
            runner.fit(
                xtrain,
                ytrain,
                xval,
                yval,
                seed=seed,
            )
            preds = runner.predict(xval)
            if task == "classification":
                classified = runner.classify(xval)
                score = metric(yval, classified)
            else:
                score = -metric(yval, preds)
            scores.append(score)

        avg = float(np.mean(scores))
        if avg > top:
            top = avg
            best = dict(params)

    if best is None:
        raise RuntimeError("No grid point evaluated.")
    return best, top
