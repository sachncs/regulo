# regulo

Pure-Python (NumPy + SciPy) reproduction of *"Adaptive Norm-Based
Regularization for Neural Networks"* by Qasim & Javed (Lund
University).  Every gradient flows through hand-written
back-propagation, every penalty exposes analytical values and
gradients, and there are no hidden deep-learning framework
defaults.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/sachncs/regulo/ci.yml?branch=master)](https://github.com/sachncs/regulo/actions)
[![PyPI](https://img.shields.io/pypi/v/regulo)](https://pypi.org/project/regulo/)

## What this is

A reference implementation of the six penalties in Qasim & Javed
(2024), the Adam optimiser, a two-hidden-layer MLP with explicit
forward / backward passes, MSE and softmax cross-entropy losses,
synthetic data generators, and K-fold cross-validation with
hyperparameter grid search.  Every numerical choice is exposed,
every gradient is auditable, and every run is reproducible with a
single integer seed.

## What this is NOT

* Not a production-grade deep learning framework.  See
  [docs/limits.md](docs/limits.md).
* Not GPU-accelerated.  All computation runs on CPU via NumPy.
* Not a hyperparameter-search framework.  Only exhaustive grid
  + K-fold is implemented.
* Not a model-deployment tool.  See
  [docs/architecture.md](docs/architecture.md) for the intended
  scope.

## Installation

```bash
pip install regulo
```

From source:

```bash
git clone https://github.com/sachncs/regulo
cd regulo
pip install -e .
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.23, SciPy ≥ 1.9.
That's it -- no scikit-learn, no pandas.

## Quick start

```python
import numpy as np
from regulo import (
    Adam, MLP, Ridge, Runner, Square, synth, Mse,
)

x, y = synth(n=200, p=20, k=10, rho=0.25, noise=0.10, seed=0)
rng = np.random.default_rng(0)
perm = rng.permutation(200)
xtrain, xtest = x[perm[:150]], x[perm[150:]]
ytrain, ytest = y[perm[:150]], y[perm[150:]]

runner = Runner(
    MLP([20, 64, 32, 1], seed=0),
    Square(),
    Ridge(lam=0.01),
    Adam(lr=1e-3),
    batch=32,
    epochs=500,
)
runner.fit(xtrain, ytrain, seed=0)
print("test MSE:", Mse()(ytest, runner.predict(xtest)))
```

## Hyperparameter search

```python
from regulo import (
    Adam, MLP, Ridge, Runner, Scalar, Square, search, synth,
)

x, y = synth(n=200, p=20, k=10, rho=0.25, noise=0.10, seed=0)
best, score = search(
    x, y,
    shape=[20, 64, 32, 1],
    method="ridge",
    grid=[{"lam": v} for v in (1e-3, 1e-2, 1e-1, 0.5)],
    loss_fn=Square(),
    folds=5,
    epochs=200,
    seed=0,
)
print("best lam:", best, "score:", score)
```

## Reproducing a paper table

See [docs/repro.md](docs/repro.md) for the full recipe.  The
bundled demo runs all six penalties across multiple DGPs and
replications:

```bash
python demo/run_simulation.py --seed 0           # 5 reps, ~1 minute
python demo/run_simulation.py --seed 0 --full    # 100 reps
```

## Six penalties

| Class      | Penalty                                                        | Hyperparameters                |
|------------|----------------------------------------------------------------|--------------------------------|
| ``Void``   | ``0``                                                          | none                           |
| ``Ridge``  | ``lam ||W||_F^2``                                              | ``lam``                        |
| ``Lasso``  | ``gamma ||W||_1``                                              | ``gamma``                      |
| ``ElasticNet`` | ``alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2``           | ``alpha``, ``gamma``           |
| ``Covridge`` | ``lambda1 ||C^{1/2} W||_F^2 + lambda2 ||W||_F^2``            | ``lambda1``, ``lambda2``, ``gram`` |
| ``Sparridge`` | ``lambda1 ||C^{1/2} W||_F^2 + gamma ||W||_1``                | ``lambda1``, ``gamma``, ``gram``  |

``Covridge`` and ``Sparridge`` apply only to the first weight
matrix (where the empirical Gram matrix is defined).

## Module layout

```
regulo/
├── __init__.py     # public API re-exports + __version__
├── penalty.py      # Penalty ABC + six concrete penalties
├── loss.py         # Loss ABC + Square, Softmax
├── net.py          # MLP, xavier
├── adam.py         # Adam optimizer
├── score.py        # Metric ABC + Mse, Mae, Rmse, R2, Balanced
├── data.py         # equicorr, synth
├── tune.py         # kfold, Scaler, resolve, search
├── store.py        # save, load (npz + json, no pickle)
└── fit.py          # Runner
```

See [docs/architecture.md](docs/architecture.md) for design
decisions and [docs/api.md](docs/api.md) for the full reference.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ --cov=regulo --cov-fail-under=95 --doctest-modules
```

The CI matrix spans Linux, macOS, and Windows across Python
3.10-3.13.  Coverage is enforced at 95%.

## Documentation

* [docs/methods.md](docs/methods.md) -- methods summary suitable
  for a paper appendix.
* [docs/math.md](docs/math.md) -- equation-by-equation mapping to
  the paper.
* [docs/repro.md](docs/repro.md) -- step-by-step reproduction
  recipe.
* [docs/limits.md](docs/limits.md) -- explicit limitations.
* [docs/architecture.md](docs/architecture.md) -- module map and
  design rationale.
* [docs/api.md](docs/api.md) -- complete API reference.

## License

[MIT](LICENSE) © 2026 Sachin

## Citation

Qasim, M. & Javed, F.  "Adaptive Norm-Based Regularization for
Neural Networks."  Lund University.
