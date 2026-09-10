# Mathematical Foundations

This document restates the core mathematics from the paper and
maps each equation to its implementation in regulo.

## Notation

* Bold lowercase: vectors ``x in R^p``
* Bold uppercase: matrices ``W in R^{p x q}``
* ``||x||_q``: ``q``-norm
* ``||A||_F``: Frobenius norm

## Regularized objective

```
J_tilde(theta; X, y) = J(theta) + Omega(W)
```

where ``J`` is the empirical loss (MSE or cross-entropy) and
``Omega`` is a penalty on the weight matrices.

**Implementation:** :class:`regulo.fit.Runner` adds
``penalty.value(W, layer)`` to the data loss on every applicable
layer at each mini-batch step.

## Standard penalties

### Ridge

```
Omega(W) = ||W||_F^2 = sum_{i,j} w_{i,j}^2
```

Gradient: ``grad Omega = 2 W``.

**Implementation:** :class:`regulo.penalty.Ridge`.

### Lasso

```
Omega(W) = ||W||_1 = sum_{i,j} |w_{i,j}|
```

Subgradient: ``grad Omega = sign(W)`` (with ``sign(0) := 0``).

**Implementation:** :class:`regulo.penalty.Lasso`.

### Elastic Net

```
Omega(W) = alpha gamma ||W||_1 + (1 - alpha)/2 ||W||_F^2
```

Gradient: ``grad Omega = alpha gamma sign(W) + (1 - alpha) W``.

**Implementation:** :class:`regulo.penalty.ElasticNet`.

## Geometry-aware penalties

### Empirical Gram matrix

```
C_n = (1/n) X^T X
C_{delta, n} = C_n + delta I_p,   delta > 0
```

**Implementation:** :func:`regulo.tune.resolve` computes
``(xtrain.T @ xtrain) / n + delta * I_p`` from the training data.

### Covridge

```
Omega(W) = lambda1 ||C^{1/2} W||_F^2 + lambda2 ||W||_F^2
```

Using the spectral decomposition ``C = U Lambda U^T``, the
penalty becomes:

```
Omega(W) = sum_{i=1}^{p} (lambda1 (mu_i + delta) + lambda2) ||w_tilde_i||_2^2
```

where ``w_tilde_i`` is the ``i``-th row of ``U^T W``.

Gradient: ``grad Omega = 2 lambda1 C W + 2 lambda2 W``.

**Implementation:** :class:`regulo.penalty.Covridge`.
``csqrt`` is precomputed via symmetric eigendecomposition.
``value`` computes ``||C^{1/2} W||_F^2`` directly.
``grad`` computes ``2 lambda1 C W + 2 lambda2 W`` using
``C = csqrt @ csqrt``.

### Sparridge

```
Omega(W) = lambda1 ||C^{1/2} W||_F^2 + gamma ||W||_1
```

Gradient (subgradient):
``grad Omega = 2 lambda1 C W + gamma sign(W)``.

**Implementation:** :class:`regulo.penalty.Sparridge`.

## Network forward pass

For a feedforward ReLU network with two hidden layers:

```
z^{(1)} = W^{(1)} x + b^{(1)}
h^{(1)} = ReLU(z^{(1)})
z^{(2)} = W^{(2)} h^{(1)} + b^{(2)}
h^{(2)} = ReLU(z^{(2)})
y_hat   = W^{(3)} h^{(2)} + b^{(3)}
```

**Implementation:** :meth:`regulo.net.MLP.__call__`.

## Back-propagation

```
dJ/dW^{(l)} = (1/n) (h^{(l-1)})^T delta^{(l)}
dJ/db^{(l)} = (1/n) sum_i delta_i^{(l)}
delta^{(l-1)} = (delta^{(l)} (W^{(l)})^T) * Indicator(z^{(l-1)} > 0)
```

**Implementation:** :meth:`regulo.net.MLP.grad`.  The ``1/n``
factor lives in the loss backward pass
(:meth:`regulo.loss.Square.grad`,
:meth:`regulo.loss.Softmax.grad`).

## Adam optimizer

```
m_t = beta1 m_{t-1} + (1 - beta1) g_t
v_t = beta2 v_{t-1} + (1 - beta2) g_t^2
mhat_t = m_t / (1 - beta1^t)
vhat_t = v_t / (1 - beta2^t)
theta_{t+1} = theta_t - lr * mhat_t / (sqrt(vhat_t) + epsilon)
```

**Implementation:** :class:`regulo.adam.Adam`.

## Loss functions

### Mean squared error

```
J_MSE = (1/N) sum_i (y_hat_i - y_i)^2
dJ/dy_hat = (2/N) (y_hat - y)
```

where ``N = y_hat.size`` (the total scalar element count).

**Implementation:** :class:`regulo.loss.Square`.

### Softmax cross-entropy

```
J_CE = -(1/n) sum_i log p_{i, y_i}
where p_{i, c} = exp(z_{i, c}) / sum_{c'} exp(z_{i, c'})
dJ/dz = (1/n) (P - Y_onehot)
```

The softmax subtracts the row-max logit before exponentiation
(numerically stable) and probabilities are clipped at ``1e-15``
before the log so confident-correct predictions yield exactly
``0.0`` loss.

**Implementation:** :class:`regulo.loss.Softmax`.

## Data generating processes

### Covariance structure

Informative predictors ``x_{1:k}`` are drawn from
``N(0, Sigma)`` where ``Sigma`` has ``Sigma_{i,i} = 1`` and
``Sigma_{i,j} = rho`` for ``i != j``.  Noise predictors
``x_{k+1:p}`` are i.i.d. standard normal.

**Implementation:** :func:`regulo.data.equicorr` builds ``Sigma``.

### Response models

Linear:
```
y = sum_{j=1}^{k} theta*_j x_j + epsilon,  epsilon ~ N(0, sigma^2)
```

Nonlinear:
```
y = sum_{j=1}^{k} theta*_j sin(x_j) + epsilon
```

**Implementation:** :func:`regulo.data.synth`.

## Theoretical results

### Theorem 5.1 (Covridge asymptotic normality)

Under fixed-design assumptions A1-A3, the Covridge estimator is
asymptotically Gaussian with sandwich covariance involving ``Q``,
``C_delta``, and the tuning parameters.

### Theorem 5.2 (Sparridge convergence)

Under assumptions A1-A4, the scaled estimation error converges to
the minimizer of a random convex criterion.  When ``gamma = 0``
the limit is Gaussian; when ``gamma > 0`` the limit is non-Gaussian.

These theorems describe asymptotic statistical properties and are
not implemented in this reproduction.
