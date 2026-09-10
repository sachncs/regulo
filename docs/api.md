# API Reference

## Penalties (`regulo.penalty`)

### `class Penalty(ABC)`

Abstract base class for weight-matrix penalties.

#### `value(weight, layer) -> float`

Scalar penalty ``Omega(W)``.

#### `grad(weight, layer) -> np.ndarray`

Analytical gradient, same shape as ``weight``.

#### `applies(layer) -> bool`

Return whether this penalty applies to ``layer``.  Default
``True`` for every layer.  :class:`Covridge` and
:class:`Sparridge` override to ``layer == 0``.

---

### `class Void(Penalty)`

No-op penalty.  Always returns ``0.0`` and zero gradient.

---

### `class Ridge(Penalty)`

**Constructor:** `Ridge(lam: float)`

Penalty: ``lam ||W||_F^2``.  Gradient: ``2 lam W``.

---

### `class Lasso(Penalty)`

**Constructor:** `Lasso(gamma: float)`

Penalty: ``gamma ||W||_1``.  Subgradient: ``gamma sign(W)``
(``sign(0) = 0``).

---

### `class ElasticNet(Penalty)`

**Constructor:** `ElasticNet(alpha: float, gamma: float)`

Penalty: ``alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2``.
Gradient: ``alpha gamma sign(W) + (1 - alpha) W``.

---

### `class Covridge(Penalty)`

**Constructor:**
`Covridge(lambda1: float, lambda2: float, gram: np.ndarray)`

Geometry-aware shrinkage along the eigenvectors of ``C``.  Applies
only to the first weight matrix.

Penalty:
``lambda1 ||C^{1/2} W||_F^2 + lambda2 ||W||_F^2``.

Gradient: ``2 lambda1 C W + 2 lambda2 W``.

---

### `class Sparridge(Penalty)`

**Constructor:**
`Sparridge(lambda1: float, gamma: float, gram: np.ndarray)`

Geometry-aware shrinkage plus L1 sparsity.  Applies only to the
first weight matrix.

Penalty:
``lambda1 ||C^{1/2} W||_F^2 + gamma ||W||_1``.

Subgradient: ``2 lambda1 C W + gamma sign(W)``.

---

### `REGISTRY: dict[str, type[Penalty]]`

Maps penalty names to penalty classes.

---

## Losses (`regulo.loss`)

### `class Loss(ABC)`

#### `value(prediction, target) -> float`

Scalar loss.

#### `grad(prediction, target) -> np.ndarray`

Gradient w.r.t. ``prediction``.

---

### `class Square(Loss)`

Mean squared error: ``(1/N) sum (y - y_hat)^2`` over ``N`` total
scalar elements.  Gradient: ``(2/N) (y_hat - y)``.

---

### `class Softmax(Loss)`

Numerically stable softmax cross-entropy.  Probabilities clipped
at ``1e-15`` before the log so confident-correct predictions give
exactly ``0.0`` loss.  ``target`` must be integer dtype in
``[0, n_classes)``.

---

## Network (`regulo.net`)

### `class MLP`

**Constructor:** `MLP(shape: list[int], seed: int | None = None)`

Feedforward MLP with ReLU hidden activations and a linear output.

#### `__call__(x) -> np.ndarray`

Forward pass; caches pre-activations (``MLP.pre``) and activations
(``MLP.post``).

#### `grad(dloss) -> dict`

Backward pass returning ``{"weights": [...], "biases": [...]}``.

#### `state() -> dict`

Deep copy of current parameters.

#### `load(state) -> None`

Replace parameters from a ``state`` dict.

---

### `xavier(fanin, fanout, rng) -> np.ndarray`

Sample a weight matrix from the Xavier-uniform distribution
using the supplied ``np.random.Generator``.

---

## Optimizer (`regulo.adam`)

### `class Adam`

**Constructor:**
`Adam(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)`

#### `step(params, grads) -> dict`

Perform one update; return new parameter dict.

#### `reset() -> None`

Zero the step counter and moment buffers.

---

## Metrics (`regulo.score`)

### `class Metric(ABC)`

#### `__call__(truth, pred) -> float`

### `class Mse`, `class Mae`, `class Rmse`, `class R2`, `class Balanced`

Concrete metrics.  All take ``truth`` and ``pred`` arrays.

---

## Data (`regulo.data`)

### `equicorr(k, rho) -> np.ndarray`

Build a ``k x k`` equi-correlation covariance matrix.  Validates
``rho in (-1/(k-1), 1)``.

### `synth(n, p, k, rho, noise, tau=1.0, nonlinear=False, seed=None) -> (X, y)`

Generate synthetic data.  Validates ``n > 0``, ``p > 0``,
``0 <= k <= p``, ``rho`` in PD range, ``noise >= 0``, ``tau >= 0``.

---

## Tuning (`regulo.tune`)

### `kfold(n, folds, seed=None) -> Iterator[tuple[np.ndarray, np.ndarray]]`

Yield ``(train, val)`` for ``folds`` shuffled folds.

### `class Scaler`

Per-column z-score scaler.  Methods: ``fit(x)``, ``transform(x)``,
``fittransform(x)``.

### `resolve(name, hp, xtrain, delta=1e-4) -> Penalty`

Construct a penalty from a name + hyperparameter dict + training
data.  Uses the :data:`REGISTRY`.

### `search(x, y, shape, method, grid, loss_fn, ...) -> (best, score)`

Run K-fold CV over a parameter grid.  Returns the highest-scoring
configuration.

---

## Persistence (`regulo.store`)

### `save(runner, path)`

Write weights/biases/adam buffers and a ``meta.json`` describing
the architecture and hyperparameters to ``path/``.

### `load(path) -> Runner`

Reconstruct a runner from ``path/``.  Refuses mismatched major
versions.

### `meta(runner) -> dict`

Return the metadata dictionary that would be written by
:meth:`save`.

### `snapshot(runner) -> dict`

Alias for :func:`meta`.

---

## Trainer (`regulo.fit`)

### `class Runner`

**Constructor:**
`Runner(mlp, loss, penalty, adam, batch=32, epochs=500, earlystop=False, patience=10)`

#### `fit(xtrain, ytrain, xval=None, yval=None, seed=None) -> None`

Train the network.

#### `predict(x) -> np.ndarray`

Raw forward pass; for classification returns logits.

#### `proba(x) -> np.ndarray`

Classification only.  Softmax probabilities.

#### `classify(x) -> np.ndarray`

Classification only.  Argmax class indices.

#### `reset(seed=None) -> None`

Re-initialise the network weights and zero the optimiser state.

#### `restart(path) -> None`

Replace the weights from a directory produced by :func:`save`.
