# Methods Summary

This page is a self-contained methods description suitable for a
paper appendix.  Every equation and algorithm implemented in
**regulo** is stated here, with the implementation file noted.

## Problem

Given training data ``(X, y)`` with ``X in R^{n x p}`` and a
two-hidden-layer MLP ``f(x; theta)`` with weights ``W^{(1)} in
R^{p x h_1}``, ``W^{(2)} in R^{h_1 x h_2}``, ``W^{(3)} in R^{h_2
x q}``, we minimize

```
J_tilde(theta) = L(y, f(X; theta)) + lam * Omega(W)
```

where ``L`` is MSE or softmax cross-entropy and ``Omega`` is one of
six penalties.

## Six penalties

| Class      | Penalty                                                        | Gradient                                            |
|------------|----------------------------------------------------------------|-----------------------------------------------------|
| ``Void``   | ``0``                                                          | ``0``                                               |
| ``Ridge``  | ``lam * ||W||_F^2``                                            | ``2 lam W``                                         |
| ``Lasso``  | ``gamma * ||W||_1``                                            | ``gamma * sign(W)`` (subgradient, ``sign(0) = 0``)  |
| ``ElasticNet`` | ``alpha gamma ||W||_1 + (1 - alpha)/2 * ||W||_F^2``         | ``alpha gamma sign(W) + (1 - alpha) W``             |
| ``Covridge`` | ``lambda1 ||C^{1/2} W||_F^2 + lambda2 ||W||_F^2``            | ``2 lambda1 C W + 2 lambda2 W``                     |
| ``Sparridge`` | ``lambda1 ||C^{1/2} W||_F^2 + gamma ||W||_1``                | ``2 lambda1 C W + gamma sign(W)`` (subgradient)     |

where ``C = (1/n) X^T X + delta I_p`` is the stabilized empirical
Gram matrix and ``delta > 0`` is a diagonal stabilizer (default
``1e-4``).

## Geometry-aware restriction

``C_{delta,n}`` is defined over the input dimension ``p`` and
therefore matches ``W^{(1)}`` only.  ``Covridge`` and ``Sparridge``
override :meth:`Penalty.applies` to return ``True`` only when
``layer == 0``; all other penalties apply to every layer.

## Forward pass

```
z^{(1)} = W^{(1)} x + b^{(1)}
h^{(1)} = ReLU(z^{(1)})
z^{(2)} = W^{(2)} h^{(1)} + b^{(2)}
h^{(2)} = ReLU(z^{(2)})
y_hat   = W^{(3)} h^{(2)} + b^{(3)}
```

For classification ``y_hat`` is interpreted as pre-softmax logits
and the softmax cross-entropy loss is applied at the output.

## Back-propagation

```
delta^{(3)} = dL/dy_hat
delta^{(2)} = delta^{(3)} (W^{(3)})^T * Indicator(z^{(2)} > 0)
delta^{(1)} = delta^{(2)} (W^{(2)})^T * Indicator(z^{(1)} > 0)
```

Gradients accumulate as ``dL/dW^{(l)} = (h^{(l-1)})^T delta^{(l)}``,
with the ``1/n_samples`` factor applied in the loss backward pass.

## Optimization

Adam (Kingma & Ba, 2014) with bias-corrected first and second
moment estimates.  Learning rate, ``beta_1``, ``beta_2``, and
``epsilon`` match the original paper's defaults.

## Synthetic data

The :func:`regulo.synth` generator parameterises three DGPs from
the paper:

| Regime       | ``n``  | ``p``  | ``k``  |
|--------------|--------|--------|--------|
| small        | 200    | 20     | 10     |
| medium       | 1000   | 200    | 100    |
| high-dim     | 500    | 2000   | 100    |

The first ``k`` columns are drawn from a multivariate normal with
equi-correlation ``rho``; the remaining columns are i.i.d.
standard normal.  Coefficients are drawn from ``N(0, tau)`` and
the response is linear or sinusoidal in the informative block.

## Cross-validation

:class:`regulo.tune.search` runs ``folds``-fold CV with optional
early stopping.  The validation fold is scored by either ``-MSE``
(regression) or balanced accuracy (classification); the grid
point with the highest mean fold score wins.
