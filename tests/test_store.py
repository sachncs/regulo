"""Tests for save/load round-trips and version validation."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from regulo import __version__
from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square
from regulo.net import MLP
from regulo.penalty import Ridge, Void
from regulo.store import load, meta, save, snapshot


def test_version_matches_package_metadata():
    assert __version__ == version("regulo")


def test_metafields():
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    m = meta(runner)
    assert m["version"] == __version__
    assert m["shape"] == [3, 4, 1]
    assert m["loss"] == "square"
    assert m["penalty"] == "none"
    assert m["task"] == "regression"
    assert m["adam"]["lr"] == 1e-3


def test_saveloadroundtrip(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 3))
    y = rng.standard_normal((20, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lam=0.01),
        Adam(lr=1e-2),
        epochs=3,
    )
    runner.fit(x, y, seed=0)
    weights = [w.copy() for w in runner.mlp.weights]
    biases = [b.copy() for b in runner.mlp.biases]
    clock = runner.adam.clock

    target = tmp_path / "model"
    save(runner, str(target))
    assert (target / "meta.json").exists()
    assert (target / "weights.npz").exists()
    assert (target / "biases.npz").exists()
    assert (target / "adam.npz").exists()

    loaded = load(str(target))
    assert loaded.mlp.shape == [3, 4, 1]
    for wn, wo in zip(loaded.mlp.weights, weights):
        np.testing.assert_array_equal(wn, wo)
    for bn, bo in zip(loaded.mlp.biases, biases):
        np.testing.assert_array_equal(bn, bo)
    assert loaded.adam.clock == clock
    assert loaded.adam.lr == 1e-2


def test_snapshot():
    runner = Runner(MLP([2, 3, 1]), Square(), Void(), Adam())
    target = Path("/tmp/_regulo_snapshot_test")
    save(runner, str(target))
    try:
        m = snapshot(runner)
        assert m["shape"] == [2, 3, 1]
    finally:
        import shutil
        shutil.rmtree(target, ignore_errors=True)


def test_loadrejectsversionmismatch(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "model"
    save(runner, str(target))

    # Tamper with the version field.
    metapath = target / "meta.json"
    data = json.loads(metapath.read_text())
    data["version"] = "999.0.0"
    metapath.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="version mismatch"):
        load(str(target))


def test_restartpreservesoptimizer(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 3))
    y = rng.standard_normal((16, 1))
    runner = Runner(
        MLP([3, 4, 1], seed=0),
        Square(),
        Ridge(lam=0.001),
        Adam(lr=1e-2),
        epochs=2,
    )
    runner.fit(x, y, seed=0)

    target = tmp_path / "warm"
    save(runner, str(target))

    newrunner = Runner(
        MLP([3, 4, 1], seed=1),
        Square(),
        Ridge(lam=0.001),
        Adam(lr=1e-3),
        epochs=2,
    )
    newrunner.restart(str(target))
    np.testing.assert_allclose(
        newrunner.mlp.weights[0], runner.mlp.weights[0]
    )
    assert newrunner.adam.clock == 0
    assert newrunner.adam.lr == 1e-3


def test_restartrejectsmismatch(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "m"
    save(runner, str(target))
    newrunner = Runner(MLP([3, 8, 1]), Square(), Void(), Adam())
    with pytest.raises(ValueError, match="Architecture mismatch"):
        newrunner.restart(str(target))


def test_loadmissingadam(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "no_adam"
    save(runner, str(target))
    (target / "adam.npz").unlink()
    loaded = load(str(target))
    assert loaded.adam.clock == 0
    for group in loaded.adam.mean:
        for buf in loaded.adam.mean[group]:
            assert buf is None
