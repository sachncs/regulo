"""Tests for k-fold splitting, standardization, and grid search."""

import numpy as np
import pytest

from regulo.tune import Scaler, kfold, resolve, search


def test_kfoldcount():
    folds = list(kfold(100, 5, seed=0))
    assert len(folds) == 5
    for train, val in folds:
        assert len(train) + len(val) == 100


def test_kfoldpartition():
    folds = list(kfold(50, 5, seed=0))
    valunion = np.concatenate([v for _, v in folds])
    assert sorted(valunion.tolist()) == list(range(50))


def test_kfoldreproducible():
    a = [v.tolist() for _, v in kfold(50, 5, seed=42)]
    b = [v.tolist() for _, v in kfold(50, 5, seed=42)]
    assert a == b


def test_kfoldrejects():
    with pytest.raises(ValueError):
        list(kfold(100, 1))
    with pytest.raises(ValueError):
        list(kfold(2, 5))


def test_scalerzscore():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 4)) * 3.0 + 5.0
    s = Scaler().fit(x)
    out = s.transform(x)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-10)


def test_scalerfittransform():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 3))
    s = Scaler().fit(x[:20])
    np.testing.assert_allclose(
        s.transform(x[:20]), s.fittransform(x[:20])
    )


def test_scalerconstantcolumn():
    x = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    out = Scaler().fittransform(x)
    assert np.all(np.isfinite(out))


def test_scalertransformunfit():
    s = Scaler()
    with pytest.raises(RuntimeError):
        s.transform(np.zeros((3, 2)))


def test_scalerrepr():
    s = Scaler()
    assert "unfit" in repr(s)
    s.fit(np.random.randn(10, 2))
    assert "Scaler(mean=" in repr(s)


def test_resolveridge():
    x = np.random.randn(50, 4)
    p = resolve("ridge", {"lam": 0.01}, x)
    assert p.name == "ridge"
    assert p.lam == 0.01


def test_resolvecovridgegram():
    x = np.random.randn(50, 4)
    p = resolve("covridge", {"lambda1": 0.1, "lambda2": 0.01}, x)
    assert p.lambda1 == 0.1
    assert p.lambda2 == 0.01
    assert p.csqrt.shape == (4, 4)


def test_resolveunknown():
    with pytest.raises(ValueError):
        resolve("nonsense", {}, np.zeros((10, 2)))


def test_resolveextrakeys():
    x = np.zeros((10, 3))
    with pytest.raises(ValueError):
        resolve("ridge", {"lam": 0.1, "garbage": 1.0}, x)


def test_searchbest():
    from regulo.loss import Square

    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    y = (
        x @ np.array([1.0, -1.0, 0.5, 0.0]).reshape(-1, 1)
        + rng.standard_normal((60, 1)) * 0.1
    )
    grid = [{"lam": 1e-3}, {"lam": 1e-2}]
    best, score = search(
        x,
        y,
        shape=[4, 4, 1],
        method="ridge",
        grid=grid,
        loss=Square(),
        folds=3,
        epochs=10,
    )
    assert "lam" in best
    assert score <= 0


def test_searchemptygrid():
    from regulo.loss import Square

    with pytest.raises(ValueError):
        search(
            np.zeros((10, 2)),
            np.zeros((10, 1)),
            shape=[2, 1],
            method="ridge",
            grid=[],
            loss=Square(),
        )


def test_searchreproducible():
    from regulo.loss import Square

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = rng.standard_normal((40, 1))
    grid = [{"lam": 1e-2}]
    _, score_a = search(
        x, y, [3, 4, 1], "ridge", grid, Square(), folds=3, epochs=5, seed=7
    )
    _, score_b = search(
        x, y, [3, 4, 1], "ridge", grid, Square(), folds=3, epochs=5, seed=7
    )
    assert score_a == score_b


def test_searchclassification():
    from regulo.loss import Softmax

    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = rng.integers(0, 2, size=(40,))
    grid = [{"lam": 1e-2}]
    best, score = search(
        x, y, [3, 4, 2], "ridge", grid, Softmax(), folds=3, epochs=5, seed=0
    )
    assert "lam" in best
    assert 0.0 <= score <= 1.0


def test_searchrejectsfolds():
    from regulo.loss import Square

    with pytest.raises(ValueError):
        search(
            np.zeros((10, 2)),
            np.zeros((10, 1)),
            shape=[2, 1],
            method="ridge",
            grid=[{"lam": 0.1}],
            loss=Square(),
            folds=1,
        )
