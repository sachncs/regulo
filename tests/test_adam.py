"""Comprehensive tests for the Adam optimizer."""

import numpy as np
import pytest

from regulo.adam import Adam


def test_updatesweights():
    opt = Adam(lr=1e-3)
    params = {"weights": [np.ones((2, 2))]}
    grads = {"weights": [np.ones((2, 2))]}
    next = opt.step(params, grads)
    assert not np.array_equal(next["weights"][0], params["weights"][0])


def test_momentestimates():
    opt = Adam(lr=0.1, beta1=0.9, beta2=0.999)
    params = {"weights": [np.zeros((1, 1))]}
    grads = {"weights": [np.ones((1, 1))]}
    for _ in range(1000):
        params = opt.step(params, grads)
    np.testing.assert_allclose(
        params["weights"][0], np.full((1, 1), -100.0), atol=1e-3
    )


def test_reset():
    opt = Adam()
    params = {"weights": [np.ones((2, 2))]}
    grads = {"weights": [np.ones((2, 2))]}
    opt.step(params, grads)
    assert opt.clock == 1
    opt.reset()
    assert opt.clock == 0
    assert opt.mean == {}
    assert opt.variance == {}


def test_zerogradient():
    opt = Adam(lr=1e-3)
    params = {"weights": [np.ones((2, 2))]}
    grads = {"weights": [np.zeros((2, 2))]}
    next = opt.step(params, grads)
    np.testing.assert_allclose(next["weights"][0], params["weights"][0])


def test_largegradient():
    opt = Adam(lr=1e-3)
    params = {"weights": [np.zeros((1, 1))]}
    grads = {"weights": [np.full((1, 1), 1e6)]}
    next = opt.step(params, grads)
    assert np.isfinite(next["weights"][0])


def test_multiplegroups():
    opt = Adam(lr=1e-3)
    params = {
        "weights": [np.ones((2, 2)), np.ones((2, 1))],
        "biases": [np.zeros((1, 2)), np.zeros((1, 1))],
    }
    grads = {
        "weights": [np.ones((2, 2)), np.ones((2, 1))],
        "biases": [np.ones((1, 2)), np.ones((1, 1))],
    }
    next = opt.step(params, grads)
    assert len(next["weights"]) == 2
    assert len(next["biases"]) == 2
    for i in range(2):
        assert not np.array_equal(
            next["weights"][i], params["weights"][i]
        )
        assert not np.array_equal(
            next["biases"][i], params["biases"][i]
        )


def test_rejectsinvalidbeta():
    with pytest.raises(ValueError):
        Adam(beta1=1.0)
    with pytest.raises(ValueError):
        Adam(beta2=1.5)
    with pytest.raises(ValueError):
        Adam(beta1=-0.1)


def test_rejectsnonpositivelr():
    with pytest.raises(ValueError):
        Adam(lr=0.0)
    with pytest.raises(ValueError):
        Adam(lr=-1.0)


def test_rejectsnonpositiveepsilon():
    with pytest.raises(ValueError):
        Adam(epsilon=0.0)
    with pytest.raises(ValueError):
        Adam(epsilon=-1e-8)


def test_repr():
    opt = Adam(lr=0.01, beta1=0.5)
    text = repr(opt)
    assert "lr=0.01" in text
    assert "beta1=0.5" in text
