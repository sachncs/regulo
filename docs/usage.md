# Usage Guide

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=regulo --cov-fail-under=95 --doctest-modules
```

Per-module:

```bash
pytest tests/test_penalty.py -v
pytest tests/test_net.py -v
pytest tests/test_fit.py -v
```

## Training a model manually

### Regression

```python
import numpy as np
from regulo import (
    Adam, MLP, Mse, Ridge, Runner, Scalar, Square, synth,
)

x, y = synth(n=200, p=20, k=10, rho=0.25, noise=0.10, seed=42)
rng = np.random.default_rng(42)
perm = rng.permutation(200)
xtrain, xtest = x[perm[:150]], x[perm[150:]]
ytrain, ytest = y[perm[:150]], y[perm[150:]]

scaler = Scaler().fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

runner = Runner(
    MLP([20, 64, 32, 1], seed=42),
    Square(),
    Ridge(lam=0.01),
    Adam(lr=1e-3),
    batch=32,
    epochs=500,
)
runner.fit(xtrain, ytrain, xval=xtest, yval=ytest, seed=42)
print(Mse()(ytest, runner.predict(xtest)))
```

### Classification

```python
import numpy as np
from regulo import (
    Adam, Balanced, Lasso, MLP, Runner, Scaler, Softmax,
)

rng = np.random.default_rng(0)
x = rng.standard_normal((200, 10))
y = rng.integers(0, 3, size=(200,))

scaler = Scaler().fit(x)
x = scaler.transform(x)

runner = Runner(
    MLP([10, 64, 32, 3], seed=0),
    Softmax(),
    Lasso(gamma=0.01),
    Adam(lr=1e-3),
    batch=32,
    epochs=200,
)
runner.fit(x, y, seed=0)
preds = runner.classify(x)
print("balanced accuracy:", Balanced()(y, preds))
```

## Hyperparameter search

```python
from regulo import search

best, score = search(
    x, y,
    shape=[10, 64, 32, 3],
    method="lasso",
    grid=[{"gamma": v} for v in (1e-3, 1e-2, 1e-1)],
    loss_fn=Softmax(),
    folds=5,
    epochs=200,
    seed=0,
)
```

## Save and load a trained model

```python
from regulo.store import save, load

save(runner, "model_dir")
restored = load("model_dir")
```

The on-disk format is plain ``npz`` + ``json`` (no ``pickle``);
loading cannot execute arbitrary code.  Major-version mismatches
are rejected.
