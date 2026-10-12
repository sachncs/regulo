"""Model serialization via NPZ + JSON.

Save and load :class:`regulo.fit.Runner` state to a directory of
NPZ and JSON files.  No ``pickle`` is used; the on-disk format
cannot execute arbitrary code on load.  The schema is stable and
versioned against :data:`regulo.__version__`.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from regulo import __version__
from regulo.adam import Adam
from regulo.contracts import SerializationError
from regulo.fit import Runner
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import REGISTRY, Penalty

__all__ = ["load", "meta", "save", "snapshot"]
SCHEMA_VERSION = 1


def meta(runner: Runner) -> dict:
    """Build the metadata dictionary describing *runner*."""

    return {
        "schema": SCHEMA_VERSION,
        "version": __version__,
        "shape": list(runner.mlp.shape),
        "loss": runner.loss.name,
        "lossargs": lossargs(runner.loss),
        "penalty": runner.penalty.name,
        "penaltyargs": penaltyargs(runner.penalty),
        "adam": {
            "lr": runner.adam.lr,
            "beta1": runner.adam.beta1,
            "beta2": runner.adam.beta2,
            "epsilon": runner.adam.epsilon,
        },
        "task": runner.task,
    }


def save(runner: Runner, path: str) -> None:
    """Persist a Runner using a recoverable directory swap."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        _save_to_directory(runner, temporary)
        backup = target.with_name(f".{target.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup.exists():
            shutil.rmtree(backup)
        _write_checksums(target)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _save_to_directory(runner: Runner, path: Path) -> None:
    """Persist a Runner to *path*.

    Creates a directory *path* containing:
      * ``meta.json`` -- architecture and hyperparameters
      * ``weights.npz`` -- one ``w{i}`` array per weight matrix
      * ``biases.npz`` -- one ``b{i}`` array per bias vector
      * ``adam.npz`` -- ``mean`` / ``variance`` / ``clock`` per group
      * ``gram.npy`` -- the geometry-aware Gram matrix (when the
        penalty is Covridge or Sparridge)
    """
    p = path
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "meta.json", "w") as f:
        json.dump(meta(runner), f, indent=2, default=serialize)
    np.savez(  # type: ignore[arg-type]
        p / "weights.npz",
        **{f"w{i}": w for i, w in enumerate(runner.mlp.weights)},  # type: ignore[arg-type]
    )
    np.savez(  # type: ignore[arg-type]
        p / "biases.npz",
        **{f"b{i}": b for i, b in enumerate(runner.mlp.biases)},  # type: ignore[arg-type]
    )
    data: dict[str, np.ndarray] = {}
    for group in runner.adam.mean:
        for i, buf in enumerate(runner.adam.mean[group]):
            data[f"{group}_{i}_mean"] = (
                np.zeros_like(runner.mlp.weights[i]) if buf is None else buf
            )
        for i, buf in enumerate(runner.adam.variance[group]):
            data[f"{group}_{i}_variance"] = (
                np.zeros_like(runner.mlp.weights[i]) if buf is None else buf
            )
    data["clock"] = np.array(runner.adam.clock)
    np.savez(p / "adam.npz", **data)  # type: ignore[arg-type]

    geom = getattr(runner.penalty, "csqrt", None)
    if geom is not None:
        gram = geom @ geom
        np.save(p / "gram.npy", gram)


def snapshot(runner: Runner) -> dict:
    """Alias for :func:`meta`.  Return the metadata dictionary."""
    return meta(runner)


def load(path: str) -> Runner:
    """Reconstruct a Runner from *path*."""
    p = Path(path)
    with open(p / "meta.json") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA_VERSION:
        raise SerializationError(
            f"Unsupported model schema: {data.get('schema')!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    major = data["version"].split(".")[0]
    lib = __version__.split(".")[0]
    if major != lib:
        raise SerializationError(
            f"version mismatch: on-disk {data['version']!r}, library {__version__!r}."
        )
    _verify_checksums(p)
    mlp = MLP(data["shape"])
    cls = losslookup(data["loss"])
    loss = cls(**data["lossargs"])
    penalty = penaltylookup(data, p)
    adam = Adam(**data["adam"])

    with np.load(p / "weights.npz", allow_pickle=False) as weights:
        for i, w in enumerate(mlp.weights):
            if f"w{i}" not in weights.files:
                raise SerializationError(f"Missing weight array w{i}.")
            loaded = weights[f"w{i}"]
            if loaded.shape != w.shape or not np.all(np.isfinite(loaded)):
                raise SerializationError(f"Invalid weight array w{i}.")
            w[...] = loaded
    with np.load(p / "biases.npz", allow_pickle=False) as biases:
        for i, b in enumerate(mlp.biases):
            if f"b{i}" not in biases.files:
                raise SerializationError(f"Missing bias array b{i}.")
            loaded = biases[f"b{i}"]
            if loaded.shape != b.shape or not np.all(np.isfinite(loaded)):
                raise SerializationError(f"Invalid bias array b{i}.")
            b[...] = loaded

    runner = Runner(mlp, loss, penalty, adam)

    # Restore Adam state if compatible.
    adampath = p / "adam.npz"
    if adampath.exists():
        with np.load(adampath, allow_pickle=False) as loaded:
            runner.adam.clock = int(loaded["clock"])
            for group in runner.adam.mean:
                for i in range(len(runner.adam.mean[group])):
                    keymean = f"{group}_{i}_mean"
                    keyvar = f"{group}_{i}_variance"
                    if keymean in loaded.files:
                        runner.adam.mean[group][i] = loaded[keymean]
                        runner.adam.variance[group][i] = loaded[keyvar]
                    else:
                        runner.adam.mean[group][i] = np.zeros_like(
                            runner.mlp.weights[i]
                            if group == "weights"
                            else runner.mlp.biases[i]
                        )
                        runner.adam.variance[group][i] = np.zeros_like(
                            runner.mlp.weights[i]
                            if group == "weights"
                            else runner.mlp.biases[i]
                        )
    return runner


def _write_checksums(path: Path) -> None:
    """Write SHA-256 checksums for all model payload files."""
    checksums = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file() and item.name != "checksums.json"
    }
    (path / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )


def _verify_checksums(path: Path) -> None:
    """Reject modified checkpoint payloads with an actionable error."""
    checksum_path = path / "checksums.json"
    if not checksum_path.exists():
        return
    try:
        checksums = json.loads(checksum_path.read_text())
        for name, expected in checksums.items():
            if name == "adam.npz" and not (path / name).exists():
                continue
            actual = hashlib.sha256((path / name).read_bytes()).hexdigest()
            if actual != expected:
                raise SerializationError(
                    f"Checkpoint checksum mismatch for {name!r}."
                )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise SerializationError("Invalid checkpoint checksums.") from exc


def lossargs(loss: Loss) -> dict:
    return {}


def penaltyargs(penalty: Penalty) -> dict:
    out = {}
    for name in penalty.hp:
        if name == "gram":
            continue
        if hasattr(penalty, name):
            out[name] = getattr(penalty, name)
    return out


def losslookup(name: str) -> type[Loss]:
    from regulo.loss import Softmax, Square

    table = {"square": Square, "softmax": Softmax}
    if name not in table:
        raise ValueError(f"Unknown loss in meta: {name!r}")
    return table[name]


def penaltylookup(data: dict, path: Path) -> Penalty:
    name = data["penalty"]
    hp = dict(data["penaltyargs"])
    if name in ("covridge", "sparridge"):
        gram_path = path / "gram.npy"
        if not gram_path.exists():
            raise ValueError(
                f"Model trained with {name!r} but the saved "
                "directory has no gram.npy.  Re-train with the "
                "current regulo version or supply a restore_gram=False "
                "load path."
            )
        hp["gram"] = np.load(gram_path, allow_pickle=False)
    cls = REGISTRY[name]
    return cls(**hp)


def serialize(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )
