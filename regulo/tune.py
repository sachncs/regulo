"""Cross-validation and hyperparameter grid search utilities.

Provides primitives for k-fold splitting, feature standardization,
penalty construction from method-name dispatch, and exhaustive
grid search over hyperparameter dictionaries.
"""

from collections.abc import Iterator

import numpy as np

from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import REGISTRY, Penalty

__all__ = ["Scaler", "kfold", "resolve", "search"]


def kfold(
    n: int, folds: int, seed: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Scaler":
        """Compute per-column mean and standard deviation."""
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must be a non-empty 2D array.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values.")
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        # Guard against constant columns (std == 0) so transform does
        # not produce NaN.
        if self.std is not None:
            floor = np.sqrt(np.finfo(float).eps)
            self.std = np.where(self.std <= floor, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fit.")
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError("x has an incompatible feature shape.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values.")
        return (x - self.mean) / self.std

    def fittransform(self, x: np.ndarray) -> np.ndarray:
        """Fit on *x* and return the transformed array."""
        self.fit(x)
        return self.transform(x)

    def __repr__(self) -> str:
        if self.mean is None:
            return "Scaler(unfit)"
        std = self.std
        if std is None:
            raise RuntimeError(
                "Scaler has no standard deviation after fitting."
            )
        return f"Scaler(mean={self.mean.tolist()!r}, std={std.tolist()!r})"


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
    if xtrain.ndim != 2 or xtrain.shape[0] == 0:
        raise ValueError("xtrain must be a non-empty 2D array.")
    if not np.all(np.isfinite(xtrain)):
        raise ValueError("xtrain must contain only finite values.")
    if delta < 0:
        raise ValueError("delta must be non-negative.")
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
    shape: list[int],
    method: str,
    grid: list[dict],
    loss: Loss,
    folds: int = 5,
    batch: int = 32,
    epochs: int = 500,
    lr: float = 1e-3,
    earlystop: bool = False,
    patience: int = 10,
    seed: int | None = None,
) -> tuple[dict | None, float]:
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
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must be a non-empty 2D array.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same sample count.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values.")
    if len(shape) < 2 or shape[0] != x.shape[1]:
        raise ValueError("shape must start with the input feature count.")
    if not grid:
        raise ValueError("grid must contain at least one entry.")
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if loss.task not in {"regression", "classification"}:
        raise ValueError("loss.task must be regression or classification.")
    task = loss.task
    top = -float("inf")
    best: dict | None = None
    from regulo.score import Balanced, Mse

    metric = Balanced() if task == "classification" else Mse()

    for params in grid:
        scores: list[float] = []
        for fold_index, (train, val) in enumerate(
            kfold(x.shape[0], folds, seed=seed)
        ):
            fold_seed = None if seed is None else seed + fold_index
            xtrain = x[train]
            xval = x[val]
            ytrain = y[train]
            yval = y[val]

            scaler = Scaler().fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xval = scaler.transform(xval)

            penalty = resolve(method, params, xtrain)
            mlp = MLP(shape, seed=fold_seed)
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
                seed=fold_seed,
            )
            preds = runner.predict(xval)
            if task == "classification":
                if preds.ndim != 2 or preds.shape[1] != shape[-1]:
                    raise ValueError("classification output shape is invalid.")
                classified = runner.classify(xval)
                score = metric(yval, classified)
            else:
                if preds.shape != yval.shape:
                    raise ValueError(
                        "regression predictions must match validation targets."
                    )
                score = -metric(yval, preds)
            if not np.isfinite(score):
                continue
            scores.append(score)

        if not scores:
            continue
        avg = float(np.nanmean(scores))
        if avg > top:
            top = avg
            best = dict(params)

    if best is None:
        raise RuntimeError("No grid point evaluated.")
    return best, top
