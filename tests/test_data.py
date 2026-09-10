"""Comprehensive tests for synthetic DGP generation."""

import numpy as np
import pytest

from regulo.data import equicorr, synth


def test_synthshapesmall():
    x, y = synth(200, 20, 10, 0.25, 0.10, seed=0)
    assert x.shape == (200, 20)
    assert y.shape == (200, 1)


def test_synthshapemid():
    x, y = synth(1000, 200, 100, 0.25, 0.10, seed=0)
    assert x.shape == (1000, 200)
    assert y.shape == (1000, 1)


def test_synthshapewide():
    x, y = synth(500, 2000, 100, 0.25, 0.10, seed=0)
    assert x.shape == (500, 2000)
    assert y.shape == (500, 1)


def test_synthcorrelation():
    rho = 0.75
    x, _ = synth(500, 20, 10, rho, 0.1, seed=42)
    empcorr = np.corrcoef(x[:, :10], rowvar=False)
    offdiag = empcorr[np.triu_indices_from(empcorr, k=1)]
    assert np.mean(offdiag) > 0.5


def test_synthnonlinear():
    x, y_lin = synth(100, 10, 5, 0.25, 0.1, nonlinear=False, seed=1)
    _, y_non = synth(100, 10, 5, 0.25, 0.1, nonlinear=True, seed=1)
    assert not np.allclose(y_lin, y_non)


def test_synthzerocorrelation():
    x, _ = synth(500, 20, 10, 0.0, 0.1, seed=42)
    empcorr = np.corrcoef(x[:, :10], rowvar=False)
    offdiag = empcorr[np.triu_indices_from(empcorr, k=1)]
    assert np.mean(np.abs(offdiag)) < 0.1


def test_synthhighcorrelation():
    x, _ = synth(500, 20, 10, 0.95, 0.1, seed=42)
    empcorr = np.corrcoef(x[:, :10], rowvar=False)
    offdiag = empcorr[np.triu_indices_from(empcorr, k=1)]
    assert np.mean(offdiag) > 0.85


def test_synthzeronoise():
    x, y = synth(100, 10, 5, 0.25, 0.0, seed=42)
    _, y2 = synth(100, 10, 5, 0.25, 0.0, seed=42)
    np.testing.assert_allclose(y, y2)


def test_synthallinformative():
    x, y = synth(50, 10, 10, 0.25, 0.1, seed=42)
    assert x.shape == (50, 10)


def test_synthnoinformative():
    x, y = synth(50, 10, 0, 0.25, 0.1, seed=42)
    assert x.shape == (50, 10)
    assert np.std(y) > 0.0


def test_synthdifferentseeds():
    x1, y1 = synth(100, 10, 5, 0.25, 0.1, seed=1)
    x2, y2 = synth(100, 10, 5, 0.25, 0.1, seed=2)
    assert not np.allclose(x1, x2)
    assert not np.allclose(y1, y2)


def test_synthrejectskgtp():
    with pytest.raises(ValueError):
        synth(50, 5, 10, 0.25, 0.1)


def test_synthrejectsnegn():
    with pytest.raises(ValueError):
        synth(-1, 5, 2, 0.0, 0.1)


def test_synthrejectsnegp():
    with pytest.raises(ValueError):
        synth(50, -1, 2, 0.0, 0.1)


def test_synthrejectsnegsigma():
    with pytest.raises(ValueError):
        synth(50, 5, 2, 0.0, -0.1)


def test_synthrejectsnegtau():
    with pytest.raises(ValueError):
        synth(50, 5, 2, 0.0, 0.1, tau=-1.0)


def test_synthrejectsnonpdrho():
    with pytest.raises(ValueError):
        synth(50, 10, 5, 1.0, 0.1)


def test_equicorrshape():
    sigma = equicorr(5, 0.3)
    assert sigma.shape == (5, 5)
    np.testing.assert_allclose(np.diag(sigma), 1.0)
    sigmaoff = sigma.copy()
    np.fill_diagonal(sigmaoff, 0.3)
    np.testing.assert_allclose(sigmaoff, 0.3)


def test_equicorrkone():
    sigma = equicorr(1, 5.0)
    np.testing.assert_allclose(sigma, [[1.0]])


def test_equicorrrejectsnonpd():
    with pytest.raises(ValueError):
        equicorr(3, 1.0)
    with pytest.raises(ValueError):
        equicorr(3, -1.0)


def test_equicorrrejectsnonpositivek():
    with pytest.raises(ValueError):
        equicorr(0, 0.0)
    with pytest.raises(ValueError):
        equicorr(-1, 0.0)
