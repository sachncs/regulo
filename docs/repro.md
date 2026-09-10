# Reproduction recipe

This page walks through reproducing one of the paper's simulation
experiments using **regulo**.

## Install

```bash
git clone https://github.com/sachncs/regulo
cd regulo
pip install -e .
```

## Single-run quick check

```python
import numpy as np
from regulo import (
    Adam, ElasticNet, Mse, MLP, Runner, Square, synth,
)

x, y = synth(n=200, p=20, k=10, rho=0.25, noise=0.10, seed=0)

rng = np.random.default_rng(0)
perm = rng.permutation(200)
xtrain, xtest = x[perm[:150]], x[perm[150:]]
ytrain, ytest = y[perm[:150]], y[perm[150:]]

runner = Runner(
    MLP([20, 64, 32, 1], seed=0),
    Square(),
    ElasticNet(alpha=0.5, gamma=0.01),
    Adam(lr=1e-3),
    batch=32,
    epochs=500,
)
runner.fit(xtrain, ytrain, seed=0)
print("test MSE:", Mse()(ytest, runner.predict(xtest)))
```

## Monte Carlo replications

The bundled demo script runs ``reps`` replications across all six
penalties and prints MSE mean / standard deviation:

```bash
python demo/run_simulation.py --seed 0           # 5 reps, ~1 minute
python demo/run_simulation.py --seed 0 --full    # 100 reps, slower
```

The demo uses a reduced hyperparameter grid for speed.  For the
full ``{0.001, 0.01, 0.1, 0.5, 0.9}`` grid from the paper, edit
``GRID`` in ``demo/run_simulation.py``.

## Save / load a trained model

```python
from regulo.store import save, load

save(runner, "model_dir")
restored = load("model_dir")
```

The on-disk format is plain ``npz`` + ``json`` (no ``pickle``), so
loading cannot execute arbitrary code.

## Cross-validation

```python
from regulo import (
    Adam, MLP, Ridge, Runner, Scalar, Square, search,
)

best, score = search(
    xtrain, ytrain,
    shape=[20, 64, 32, 1],
    method="ridge",
    grid=[{"lam": v} for v in (1e-3, 1e-2, 1e-1)],
    loss_fn=Square(),
    folds=5,
    epochs=200,
    seed=0,
)
```

``score`` is ``-MSE`` (higher is better) for regression and
balanced accuracy for classification.
