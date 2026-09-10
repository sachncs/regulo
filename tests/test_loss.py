"""Comprehensive tests for loss functions."""

import numpy as np
import pytest

from regulo.loss import Loss, Softmax, Square


def test_lossabstract():
    with pytest.raises(TypeError):
        Loss()


def test_squareperfect():
    y = np.array([[1.0], [2.0], [3.0]])
    loss = Square()
    assert loss.value(y, y) == 0.0
    np.testing.assert_allclose(loss.grad(y, y), 0.0)


def test_squareknown():
    ypred = np.array([[0.0], [2.0]])
    truth = np.array([[1.0], [1.0]])
    loss = Square()
    assert loss.value(ypred, truth) == 1.0


def test_squaregradshape():
    ypred = np.random.randn(8, 3)
    truth = np.random.randn(8, 3)
    loss = Square()
    grad = loss.grad(ypred, truth)
    assert grad.shape == (8, 3)


def test_squaresingle():
    ypred = np.array([[5.0]])
    truth = np.array([[3.0]])
    loss = Square()
    assert loss.value(ypred, truth) == 4.0
    np.testing.assert_allclose(loss.grad(ypred, truth), [[4.0]])


def test_softmaxknown():
    logits = np.array([[1.0, 0.0]])
    target = np.array([0])
    loss = Softmax()
    loss = loss.value(logits, target)
    expected = float(np.log(1 + np.exp(-1)))
    assert np.isclose(loss, expected)


def test_softmaxperfect():
    logits = np.array([[1e6, -1e6, -1e6]])
    target = np.array([0])
    loss = Softmax()
    assert loss.value(logits, target) == 0.0


def test_softmaxgradshape():
    logits = np.random.randn(8, 3)
    target = np.random.randint(0, 3, size=(8,))
    loss = Softmax()
    grad = loss.grad(logits, target)
    assert grad.shape == (8, 3)


def test_softmaxgradsumzero():
    logits = np.random.randn(8, 3)
    target = np.random.randint(0, 3, size=(8,))
    grad = Softmax().grad(logits, target)
    np.testing.assert_allclose(np.sum(grad, axis=1), 0.0, atol=1e-12)


def test_softmaxrejectsfloat():
    logits = np.array([[1.0, 0.0]])
    target = np.array([0.5])
    loss = Softmax()
    with pytest.raises(TypeError):
        loss.value(logits, target)


def test_softmaxrejectsoutofrange():
    logits = np.array([[1.0, 0.0]])
    target = np.array([5])
    loss = Softmax()
    with pytest.raises(ValueError):
        loss.value(logits, target)


def test_softmaxsingleclass():
    logits = np.array([[0.0], [1.0], [2.0]])
    target = np.array([0, 0, 0])
    loss = Softmax()
    assert np.isclose(loss.value(logits, target), 0.0)
    np.testing.assert_allclose(loss.grad(logits, target), 0.0)


def test_lossrepr():
    assert "Square" in repr(Square())
    assert "Softmax" in repr(Softmax())


def test_lossnames():
    assert Softmax().name == "softmax"
    assert Square().name == "square"
