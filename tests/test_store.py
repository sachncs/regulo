"""Tests for save/load round-trips and version validation."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import regulo.store as store
from regulo import __version__
from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square
from regulo.net import MLP
from regulo.penalty import Covridge, Ridge, Sparridge, Void
from regulo.store import load, meta, save, snapshot


def refresh_checksums(target: Path) -> None:
    checksums = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in target.iterdir()
        if item.is_file() and item.name != "checksums.json"
    }
    (target / "checksums.json").write_text(json.dumps(checksums))


def test_version_matches_package_metadata():
    assert __version__ == version("regulo")


def test_metafields():
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    m = meta(runner)
    assert m["schema"] == 1
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
    for wn, wo in zip(loaded.mlp.weights, weights, strict=False):
        np.testing.assert_array_equal(wn, wo)
    for bn, bo in zip(loaded.mlp.biases, biases, strict=False):
        np.testing.assert_array_equal(bn, bo)
    assert loaded.adam.clock == clock
    assert loaded.adam.lr == 1e-2
    for group in runner.adam.mean:
        for expected, actual in zip(
            runner.adam.mean[group], loaded.adam.mean[group], strict=False
        ):
            np.testing.assert_array_equal(expected, actual)
    assert (target / "checksums.json").exists()


def test_save_replaces_existing_checkpoint(tmp_path: Path):
    runner = Runner(MLP([2, 2, 1]), Square(), Void(), Adam())
    target = tmp_path / "replace"
    save(runner, str(target))
    save(runner, str(target))
    save(runner, str(target))
    assert load(str(target)).mlp.shape == [2, 2, 1]


def test_save_restores_previous_checkpoint_when_swap_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = Runner(MLP([2, 2, 1]), Square(), Void(), Adam())
    target = tmp_path / "rollback"
    save(runner, str(target))
    real_replace = store.os.replace
    calls = 0

    def fail_new_target(source: str | bytes, destination: str | bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated checkpoint swap failure")
        real_replace(source, destination)

    monkeypatch.setattr(store.os, "replace", fail_new_target)
    with pytest.raises(OSError, match="swap failure"):
        save(runner, str(target))
    assert load(str(target)).mlp.shape == [2, 2, 1]


def test_checksum_detects_corruption(tmp_path: Path):
    runner = Runner(MLP([2, 2, 1]), Square(), Void(), Adam())
    target = tmp_path / "checked"
    save(runner, str(target))
    with (target / "weights.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load(str(target))


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
    np.testing.assert_allclose(newrunner.mlp.weights[0], runner.mlp.weights[0])
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
    with pytest.raises(ValueError, match="Missing checkpoint file"):
        load(str(target))


def test_loadpartialadam_initializes_missing_buffers(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "partial_adam"
    save(runner, str(target))
    (target / "checksums.json").unlink()
    np.savez(target / "adam.npz", clock=np.array(0))
    refresh_checksums(target)
    loaded = load(str(target))
    for group in loaded.adam.mean:
        assert all(buffer is not None for buffer in loaded.adam.mean[group])


def test_load_rejects_invalid_adam_buffers(tmp_path: Path):
    runner = Runner(MLP([2, 2, 1]), Square(), Void(), Adam())
    target = tmp_path / "invalid_adam"
    save(runner, str(target))
    (target / "checksums.json").unlink()
    np.savez(
        target / "adam.npz",
        clock=np.array(0),
        weights_0_mean=np.full((1, 1), np.nan),
        weights_0_variance=np.zeros((2, 2)),
    )
    refresh_checksums(target)
    with pytest.raises(ValueError, match="Invalid Adam buffers"):
        load(str(target))


def test_load_rejects_missing_and_invalid_arrays(tmp_path: Path):
    runner = Runner(MLP([2, 2, 1]), Square(), Void(), Adam())
    missing = tmp_path / "missing"
    save(runner, str(missing))
    (missing / "checksums.json").unlink()
    np.savez(missing / "weights.npz", w0=np.zeros((2, 2)))
    refresh_checksums(missing)
    with pytest.raises(ValueError, match="Missing weight"):
        load(str(missing))

    invalid = tmp_path / "invalid"
    save(runner, str(invalid))
    (invalid / "checksums.json").unlink()
    np.savez(
        invalid / "biases.npz",
        b0=np.zeros((1, 2)),
        b1=np.full((1, 1), np.nan),
    )
    refresh_checksums(invalid)
    with pytest.raises(ValueError, match="Invalid bias"):
        load(str(invalid))


def test_saveloadpreservescovridgegram(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    gram = (x.T @ x) / 20 + 1e-4 * np.eye(4)
    penalty = Covridge(lambda1=0.01, lambda2=0.001, gram=gram)
    runner = Runner(
        MLP([4, 5, 1], seed=0),
        Square(),
        penalty,
        Adam(lr=1e-2),
        epochs=1,
    )
    runner.fit(x, rng.standard_normal((20, 1)), seed=0)

    target = tmp_path / "covridge_model"
    save(runner, str(target))
    assert (target / "gram.npy").exists()

    loaded = load(str(target))
    np.testing.assert_allclose(loaded.penalty.csqrt, penalty.csqrt)


def test_saveloadpreservessparridgegram(tmp_path: Path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    gram = (x.T @ x) / 20 + 1e-4 * np.eye(4)
    penalty = Sparridge(lambda1=0.01, gamma=0.001, gram=gram)
    runner = Runner(
        MLP([4, 5, 1], seed=0),
        Square(),
        penalty,
        Adam(lr=1e-2),
        epochs=1,
    )
    runner.fit(x, rng.standard_normal((20, 1)), seed=0)

    target = tmp_path / "sparridge_model"
    save(runner, str(target))
    assert (target / "gram.npy").exists()

    loaded = load(str(target))
    np.testing.assert_allclose(loaded.penalty.csqrt, penalty.csqrt)


def test_loadraiseswhencovridgegrammissing(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "stale"
    save(runner, str(target))
    metapath = target / "meta.json"
    data = json.loads(metapath.read_text())
    data["penalty"] = "covridge"
    data["penaltyargs"] = {"lambda1": 0.01, "lambda2": 0.001}
    metapath.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="checksum"):
        load(str(target))


def test_load_rejects_unknown_schema(tmp_path: Path):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / "schema"
    save(runner, str(target))
    data = json.loads((target / "meta.json").read_text())
    data["schema"] = 999
    (target / "meta.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        load(str(target))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", "invalid", "invalid version"),
        ("shape", [3], "invalid shape"),
        ("lossargs", [], "invalid parameters"),
    ],
)
def test_load_rejects_invalid_metadata(
    tmp_path: Path, field: str, value: object, message: str
):
    runner = Runner(MLP([3, 4, 1]), Square(), Void(), Adam())
    target = tmp_path / field
    save(runner, str(target))
    data = json.loads((target / "meta.json").read_text())
    data[field] = value
    (target / "meta.json").write_text(json.dumps(data))
    refresh_checksums(target)
    with pytest.raises(ValueError, match=message):
        load(str(target))


def test_load_rejects_missing_or_non_object_metadata(tmp_path: Path):
    missing = tmp_path / "missing_meta"
    missing.mkdir()
    with pytest.raises(ValueError, match="Invalid checkpoint metadata"):
        load(str(missing))

    non_object = tmp_path / "non_object_meta"
    non_object.mkdir()
    (non_object / "meta.json").write_text("[]")
    with pytest.raises(ValueError, match="must be an object"):
        load(str(non_object))


def test_load_rejects_invalid_adam_clock(tmp_path: Path):
    runner = Runner(MLP([2, 2, 1]), Square(), Void(), Adam())
    target = tmp_path / "invalid_clock"
    save(runner, str(target))
    (target / "checksums.json").unlink()
    np.savez(target / "adam.npz", clock=np.array([0]))
    refresh_checksums(target)
    with pytest.raises(ValueError, match="invalid Adam clock"):
        load(str(target))


def test_load_rejects_missing_bias_array(tmp_path: Path):
    runner = Runner(MLP([2, 2, 1]), Square(), Void(), Adam())
    target = tmp_path / "missing_bias"
    save(runner, str(target))
    (target / "checksums.json").unlink()
    np.savez(target / "biases.npz", b0=np.zeros((1, 2)))
    refresh_checksums(target)
    with pytest.raises(ValueError, match="Missing bias"):
        load(str(target))


def test_store_helpers_reject_unknown_values():
    from regulo.store import losslookup, serialize

    with pytest.raises(ValueError, match="Unknown loss"):
        losslookup("unknown")
    with pytest.raises(TypeError, match="not JSON serializable"):
        serialize(object())
