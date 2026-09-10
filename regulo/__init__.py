"""regulo -- Adaptive Norm-Based Regularization for Neural Networks.

Pure-Python (NumPy + SciPy) reproduction of the empirical methodology
from Qasim & Javed (Lund University).  Designed for *transparency* and
*reproducibility* rather than throughput: every gradient flows through
hand-written back-propagation, every penalty exposes analytical
penalties and gradients, and there are no hidden deep-learning
framework defaults.

Quick start
-----------
>>> from regulo import MLP, Adam, Runner, Ridge, Square
>>> net = MLP([10, 32, 1], seed=0)
>>> trainer = Runner(net, Square(), Ridge(0.01), Adam(), seed=0)
>>> trainer.fit(X_train, y_train, seed=0)
>>> preds = trainer.predict(X_test)
"""
from importlib.metadata import PackageNotFoundError, version

from regulo.adam import Adam
from regulo.data import equicorr, synth
from regulo.loss import Loss, Softmax, Square
from regulo.net import MLP, xavier
from regulo.penalty import (
    REGISTRY,
    Covridge,
    ElasticNet,
    Lasso,
    Penalty,
    Ridge,
    Sparridge,
    Void,
)
from regulo.score import Balanced, Mae, Metric, Mse, R2, Rmse
from regulo.tune import Scaler, kfold, resolve, search

try:
    __version__ = version("regulo")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# store depends on __version__, so it must be imported last.
from regulo.fit import Runner  # noqa: E402
from regulo.store import load, meta, save, snapshot  # noqa: E402

__all__ = [
    "Adam",
    "Balanced",
    "Covridge",
    "ElasticNet",
    "Lasso",
    "Loss",
    "Mae",
    "Metric",
    "MLP",
    "Mse",
    "Penalty",
    "REGISTRY",
    "R2",
    "Ridge",
    "Rmse",
    "Runner",
    "Scaler",
    "Softmax",
    "Sparridge",
    "Square",
    "Void",
    "equicorr",
    "kfold",
    "load",
    "meta",
    "resolve",
    "save",
    "snapshot",
    "search",
    "synth",
    "xavier",
]
