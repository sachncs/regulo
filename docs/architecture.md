# Architecture

## Overview

**regulo** is a pure-Python, NumPy + SciPy reproduction of the
paper's empirical methodology.  Every algorithmic component is
implemented from scratch to ensure maximum transparency and
reproducibility.

## Module map

```
regulo/
├── __init__.py     # public API re-exports + __version__
├── penalty.py      # Penalty ABC + Ridge, Lasso, ElasticNet, Covridge, Sparridge, Void
├── loss.py         # Loss ABC + Square, Softmax
├── net.py          # MLP, xavier
├── adam.py         # Adam optimizer
├── score.py        # Metric ABC + Mse, Mae, Rmse, R2, Balanced
├── data.py         # equicorr, synth (DGP generators)
├── tune.py         # kfold, Scaler, resolve, search
├── store.py        # save, load, snapshot, meta (npz + json)
└── fit.py          # Runner (training loop)
```

## Data flow

```
[Data Source]
    |
    v
[synth(...) or user-provided X, y]
    |
    v
[Scaler.fit_transform on train]
    |
    v
[Gram matrix] --> [resolve(method, hp, xtrain)]
    |
    v
[Penalty subclass]   <--  Penalty.applies(layer)
    |                       ^
    v                       |
[MLP.__call__]              |
    |                       |
    v                       |
[Runner.fit] ---------------+
    |
    +-- Mini-batch sampling
    +-- Forward pass (cache pre, post)
    +-- Loss + penalty value (only where applies)
    +-- Loss + penalty gradient (only where applies)
    +-- Adam step
    +-- NaN guard
    +-- Validation / earlystop
    |
    v
[Metrics: Mse, Mae, Rmse, R2, Balanced]
```

## Key design decisions

### 1. Pure NumPy back-propagation

No PyTorch / TensorFlow / JAX.  Every gradient flows through
hand-written code; weights are inspectable at every step.

### 2. Polymorphic Penalty dispatch

Covridge and Sparridge are applied **only to the first layer**
via the :meth:`Penalty.applies` hook.  No ``isinstance`` checks
in :class:`Runner` -- dispatch is uniformly polymorphic.

### 3. Loss-side scaling

The ``1/n_samples`` factor lives in
:meth:`regulo.loss.Square.grad` and
:meth:`regulo.loss.Softmax.grad`.  This matches standard deep
learning frameworks.

### 4. Penalty registry

:class:`regulo.penalty.REGISTRY` maps method names to penalty
classes.  :func:`regulo.tune.resolve` is the single entry point
for constructing a penalty from a string name and a hyperparameter
dictionary; no method-name dispatch lives outside the registry.

### 5. Determinism via seeds

Every stochastic call site accepts a ``seed`` integer:

* :class:`regulo.net.MLP` initialisation
* :meth:`regulo.fit.Runner.fit` mini-batch shuffling
* :func:`regulo.data.synth` DGP sampling
* :func:`regulo.tune.search` K-fold splits

A given seed produces bit-identical weights across runs.

### 6. Persistence without pickle

:class:`regulo.store.save` writes plain ``npz`` and ``json``
files.  Loading cannot execute arbitrary code; the schema is
versioned via :data:`regulo.__version__` and rejects mismatched
major versions.

### 7. Cross-validation standardisation

Each fold in :func:`regulo.tune.search` fits a fresh
:class:`regulo.tune.Scaler` on the training partition only,
preventing information leakage from the validation set.

## Extensibility

* **New penalty:** subclass :class:`Penalty`, implement
  :meth:`value` and :meth:`grad`, optionally override
  :meth:`applies`, add to :data:`REGISTRY`.
* **New loss:** subclass :class:`Loss`, implement :meth:`value`
  and :meth:`grad`.
* **New optimizer:** subclass or replace :class:`Adam` -- the
  :class:`Runner` only relies on the ``step(params, grads)``
  contract.
* **New metric:** subclass :class:`Metric`, implement
  :meth:`__call__`.
