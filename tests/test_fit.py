"""Integration tests: end-to-end training with all penalties."""

import numpy as np
import pytest

from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square, Softmax
from regulo.net import MLP
from regulo.penalty import (
    Covridge,
    ElasticNet,
    Lasso,
    Ridge,
    Sparridge,
    Void,
)
from regulo.score import Balanced, Mse


def test_fitdecreasesloss():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 5))
    y = (
        x @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]).reshape(-1, 1)
        + rng.standard_normal((64, 1)) * 0.1
    )
    runner = Runner(
        MLP([5, 8, 1]),
        Square(),
        Ridge(lam=1e-4),
        Adam(lr=1e-2),
        batch=16,
        epochs=100,
    )
    runner.fit(x, y, seed=0)
    assert runner.history["train"][-1] < runner.history["train"][0]
    assert runner.predict(x).shape == y.shape


def test_classificationshape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 4))
    y = rng.integers(0, 3, size=(32,))
    runner = Runner(
        MLP([4, 8, 3]),
        Softmax(),
        Ridge(lam=1e-4),
        Adam(lr=1e-2),
        batch=16,
        epochs=50,
    )
    runner.fit(x, y, seed=0)
    logits = runner.predict(x)
    assert logits.shape == (32, 3)
    preds = runner.classify(x)
    assert preds.shape == (32,)
    assert 0.0 <= Balanced()(y, preds) <= 1.0


def test_traineachpenalty():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 4))
    y = rng.standard_normal((40, 1))
    gram = np.eye(4)
    penalties = [
        Void(),
        Ridge(lam=1e-4),
        Lasso(gamma=1e-4),
        ElasticNet(alpha=0.5, gamma=1e-4),
        Covridge(lambda1=1e-4, lambda2=1e-4, gram=gram),
        Sparridge(lambda1=1e-4, gamma=1e-4, gram=gram),
    ]
    for penalty in penalties:
        runner = Runner(
            MLP([4, 4, 1]),
            Square(),
            penalty,
            Adam(lr=1e-2),
            batch=16,
            epochs=20,
        )
        runner.fit(x, y, seed=0)
        assert len(runner.history["train"]) == 20


def test_earlystop():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 4))
    y = rng.standard_normal((64, 1))
    runner = Runner(
        MLP([4, 4, 1]),
        Square(),
        Ridge(lam=1e-4),
        Adam(lr=1e-2),
        batch=16,
        epochs=500,
        earlystop=True,
        patience=5,
    )
    runner.fit(x[:40], y[:40], x[40:], y[40:], seed=0)
    assert len(runner.history["train"]) < 500


def test_scalerconsistency():
    from regulo.tune import Scaler

    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 4))
    y = rng.standard_normal((100, 1))
    scaler = Scaler().fit(x[:80])
    xtrain = scaler.transform(x[:80])
    xtest = scaler.transform(x[80:])
    runner = Runner(
        MLP([4, 4, 1]),
        Square(),
        Ridge(lam=1e-4),
        Adam(),
        batch=16,
        epochs=10,
    )
    runner.fit(xtrain, y[:80], seed=0)
    assert runner.predict(xtest).shape == (20, 1)


def test_historykeys():
    runner = Runner(
        MLP([3, 2, 1]),
        Square(),
        Ridge(lam=1e-4),
        Adam(),
        epochs=5,
    )
    x = np.random.randn(32, 3)
    y = np.random.randn(32, 1)
    runner.fit(x, y, seed=0)
    assert "train" in runner.history
    assert len(runner.history["train"]) == 5
    assert "val" in runner.history
    assert len(runner.history["val"]) == 0


def test_historywithval():
    runner = Runner(
        MLP([3, 2, 1]),
        Square(),
        Ridge(lam=1e-4),
        Adam(),
        epochs=5,
    )
    x = np.random.randn(32, 3)
    y = np.random.randn(32, 1)
    runner.fit(x[:20], y[:20], x[20:], y[20:], seed=0)
    assert len(runner.history["val"]) == 5


def test_taskderived():
    assert Runner(MLP([3, 1]), Square(), Void(), Adam()).task == "regression"
    assert Runner(MLP([3, 2]), Softmax(), Void(), Adam()).task == "classification"


def test_probaregressionraises():
    runner = Runner(MLP([3, 1]), Square(), Void(), Adam())
    with pytest.raises(NotImplementedError):
        runner.proba(np.zeros((1, 3)))


def test_probasumsone():
    runner = Runner(MLP([3, 4]), Softmax(), Void(), Adam(), epochs=1)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 3))
    y = rng.integers(0, 4, size=(8,))
    runner.fit(x, y, seed=0)
    proba = runner.proba(x)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_resetreinitializes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal((20, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lam=0.01),
        Adam(),
        epochs=2,
    )
    runner.fit(x, y, seed=0)
    runner.reset(seed=0)
    assert runner.history["train"] == []
    fresh = MLP([3, 4, 1], seed=0)
    for wr, wf in zip(runner.mlp.weights, fresh.weights):
        np.testing.assert_array_equal(wr, wf)
    assert runner.adam.clock == 0


def test_reproducibleseed():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = rng.standard_normal((40, 1))
    r1 = Runner(
        MLP([3, 8, 1], seed=42), Square(), Ridge(0.001), Adam(), epochs=5, batch=8
    )
    r2 = Runner(
        MLP([3, 8, 1], seed=42), Square(), Ridge(0.001), Adam(), epochs=5, batch=8
    )
    r1.fit(x, y, seed=123)
    r2.fit(x, y, seed=123)
    np.testing.assert_allclose(r1.history["train"], r2.history["train"])


def test_nanguard():
    runner = Runner(
        MLP([3, 4, 1]),
        Square(),
        Void(),
        Adam(),
        batch=4,
        epochs=2,
    )
    runner.mlp.weights[0][0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        runner.fit(np.zeros((10, 3)), np.zeros((10, 1)), seed=0)


def test_earlystoprequiresval():
    runner = Runner(
        MLP([3, 2, 1]),
        Square(),
        Ridge(lam=0.001),
        Adam(),
        epochs=2,
        earlystop=True,
    )
    with pytest.raises(ValueError):
        runner.fit(np.zeros((10, 3)), np.zeros((10, 1)))


def test_validatepositive():
    with pytest.raises(ValueError):
        Runner(MLP([3, 1]), Square(), Void(), Adam(), batch=0)
    with pytest.raises(ValueError):
        Runner(MLP([3, 1]), Square(), Void(), Adam(), epochs=0)
    with pytest.raises(ValueError):
        Runner(MLP([3, 1]), Square(), Void(), Adam(), patience=0)


def test_repr():
    runner = Runner(MLP([3, 1]), Square(), Void(), Adam())
    text = repr(runner)
    assert "Runner(" in text
    assert "MLP(" in text
    assert "Square(" in text
    assert "Adam(" in text
