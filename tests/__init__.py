"""Test suite for the ``regulo`` package.

Contains unit tests for individual modules (net, penalty, loss,
adam, score, data generators) and integration tests that exercise
the full train-evaluate pipeline through :class:`~regulo.fit.Runner`
and :func:`~regulo.tune.search`.

Test layout
-----------
* :mod:`tests.test_net` -- forward / backward shapes, single-layer
  and multi-output behaviour, gradient agreement with central finite
  differences.
* :mod:`tests.test_penalty` -- value/grad correctness for every
  penalty, parameter-validation errors, special-case reductions
  (Elastic-Net at ``alpha in {0, 1}``, identity Covridge/Sparridge).
* :mod:`tests.test_loss` -- known-value forward scores and
  gradients, numerical stability for the cross-entropy softmax.
* :mod:`tests.test_adam` -- Adam updates, bias correction,
  reset, multi-group parameter handling.
* :mod:`tests.test_score` -- MSE / MAE / RMSE / R-squared /
  balanced accuracy including constant-target and class-imbalance
  edge cases.
* :mod:`tests.test_data` -- DGP shapes, equi-correlation structure,
  zero-noise determinism, seed independence.
* :mod:`tests.test_fit` -- end-to-end training for regression and
  classification, every regularizer invoked, early stopping,
  cross-validation, NaN guard, reset, warm_start.

Running
-------
``pytest`` from the repository root discovers all tests
automatically; coverage is reported via ``pytest --cov=regulo``.
"""
