"""Comprehensive tests for net forward pass, backward pass, and gradient checks."""

import numpy as np
import pytest

from regulo.net import MLP, xavier


def test_forwardshape():
    net = MLP([10, 64, 32, 1])
    x = np.random.randn(8, 10)
    y = net(x)
    assert y.shape == (8, 1)


def test_forwardmultioutput():
    net = MLP([5, 4, 3])
    x = np.random.randn(10, 5)
    y = net(x)
    assert y.shape == (10, 3)


def test_forwardsinglesample():
    net = MLP([4, 3, 2])
    x = np.random.randn(1, 4)
    y = net(x)
    assert y.shape == (1, 2)


def test_forwardsinglelayer():
    net = MLP([3, 1])
    x = np.random.randn(5, 3)
    y = net(x)
    assert y.shape == (5, 1)


def test_forwardzeroinput():
    net = MLP([3, 2, 1])
    x = np.zeros((4, 3))
    y = net(x)
    np.testing.assert_allclose(y, 0.0, atol=1e-12)


def test_backwardshape():
    net = MLP([4, 3, 2])
    x = np.random.randn(16, 4)
    out = net(x)
    grads = net.grad(np.ones_like(out))
    assert len(grads["weights"]) == 2
    assert len(grads["biases"]) == 2
    assert grads["weights"][0].shape == (4, 3)
    assert grads["weights"][1].shape == (3, 2)
    assert grads["biases"][0].shape == (1, 3)
    assert grads["biases"][1].shape == (1, 2)


def test_backwardsinglesample():
    net = MLP([3, 4, 1])
    x = np.random.randn(1, 3)
    out = net(x)
    grads = net.grad(np.ones_like(out))
    assert grads["weights"][0].shape == (3, 4)
    assert grads["weights"][1].shape == (4, 1)


def test_backwardsinglelayer():
    net = MLP([3, 1])
    x = np.random.randn(8, 3)
    out = net(x)
    grads = net.grad(np.ones_like(out))
    assert len(grads["weights"]) == 1
    assert grads["weights"][0].shape == (3, 1)


def test_backwardzeroinput():
    net = MLP([3, 2, 1])
    x = np.zeros((4, 3))
    out = net(x)
    grads = net.grad(np.ones_like(out))
    np.testing.assert_allclose(grads["weights"][0], 0.0, atol=1e-12)


def test_stateloadroundtrip():
    net = MLP([3, 4, 2])
    state = net.state()
    state["weights"][0] = np.ones_like(state["weights"][0])
    net.load(state)
    snapshot = net.state()
    np.testing.assert_allclose(snapshot["weights"][0], np.ones_like(state["weights"][0]))


def fdgradient(net: MLP, x: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> dict:
    """Compute weight/bias gradients via central finite differences."""
    out = net(x)
    loss = float(np.mean((out - y) ** 2))
    fdw = []
    for i, w in enumerate(net.weights):
        grad = np.zeros_like(w)
        for r in range(w.shape[0]):
            for c in range(w.shape[1]):
                wplus = w.copy()
                wplus[r, c] += eps
                orig = net.weights[i].copy()
                net.weights[i] = wplus
                lossexp = float(np.mean((net(x) - y) ** 2))
                net.weights[i] = orig
                grad[r, c] = (lossexp - loss) / eps
        fdw.append(grad)
    fdb = []
    for i, b in enumerate(net.biases):
        grad = np.zeros_like(b)
        for c in range(b.shape[1]):
            wplus = b.copy()
            wplus[0, c] += eps
            orig = net.biases[i].copy()
            net.weights[i] = net.weights[i]
            net.biases[i] = wplus
            lossexp = float(np.mean((net(x) - y) ** 2))
            net.biases[i] = orig
            grad[0, c] = (lossexp - loss) / eps
        fdb.append(grad)
    return {"weights": fdw, "biases": fdb}


def test_gradientfdsingle():
    rng = np.random.default_rng(0)
    net = MLP([3, 4, 1])
    x = rng.standard_normal((4, 3))
    y = rng.standard_normal((4, 1))
    out = net(x)
    dloss = 2.0 * (out - y) / out.size
    grads = net.grad(dloss)
    fd = fdgradient(net, x, y)
    for i in range(len(grads["weights"])):
        np.testing.assert_allclose(
            grads["weights"][i], fd["weights"][i], rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            grads["biases"][i], fd["biases"][i], rtol=1e-3, atol=1e-4
        )


def test_gradientfdmulti():
    rng = np.random.default_rng(2)
    net = MLP([3, 4, 2])
    x = rng.standard_normal((5, 3))
    y = rng.standard_normal((5, 2))
    out = net(x)
    dloss = 2.0 * (out - y) / out.size
    grads = net.grad(dloss)
    fd = fdgradient(net, x, y)
    for i in range(len(grads["weights"])):
        np.testing.assert_allclose(
            grads["weights"][i], fd["weights"][i], rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            grads["biases"][i], fd["biases"][i], rtol=1e-3, atol=1e-4
        )


def test_reproducibleseed():
    a = MLP([3, 4, 1], seed=42)
    b = MLP([3, 4, 1], seed=42)
    for wa, wb in zip(a.weights, b.weights):
        np.testing.assert_array_equal(wa, wb)


def test_differentseedsdiffer():
    a = MLP([3, 4, 1], seed=0)
    b = MLP([3, 4, 1], seed=1)
    assert not np.array_equal(a.weights[0], b.weights[0])


def test_rejectstoofewlayers():
    with pytest.raises(ValueError):
        MLP([3])


def test_rejectsnonpositivewidth():
    with pytest.raises(ValueError):
        MLP([3, 0, 1])
    with pytest.raises(ValueError):
        MLP([-1, 3, 1])


def test_gradbeforecallraises():
    net = MLP([3, 1])
    with pytest.raises(RuntimeError):
        net.grad(np.ones((1, 1)))


def test_xaviershapebound():
    rng = np.random.default_rng(0)
    w = xavier(3, 4, rng)
    assert w.shape == (3, 4)
    assert np.max(np.abs(w)) <= np.sqrt(6.0 / 7) + 1e-12


def test_xavierreproducible():
    a = xavier(3, 4, np.random.default_rng(0))
    b = xavier(3, 4, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)


def test_repr():
    assert repr(MLP([3, 4, 1])) == "MLP(shape=[3, 4, 1])"
