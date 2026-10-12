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
    """Persist a Runner using a crash-safe directory swap."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    backup = target.with_name(f".{target.name}.previous")
    try:
        _save_to_directory(runner, temporary)
        _write_checksums(temporary)
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if not target.exists() and backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
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
    try:
        with open(p / "meta.json") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise SerializationError("Invalid checkpoint metadata.") from exc
    if not isinstance(data, dict):
        raise SerializationError("Checkpoint metadata must be an object.")
    if data.get("schema") != SCHEMA_VERSION:
        raise SerializationError(
            f"Unsupported model schema: {data.get('schema')!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    version = data.get("version")
    if not isinstance(version, str) or not version or "." not in version:
        raise SerializationError("Checkpoint metadata has an invalid version.")
    major = version.split(".")[0]
    lib = __version__.split(".")[0]
    if major != lib:
        raise SerializationError(
            f"version mismatch: on-disk {version!r}, library {__version__!r}."
        )
    shape = data.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or any(not isinstance(width, int) or width <= 0 for width in shape)
    ):
        raise SerializationError("Checkpoint metadata has an invalid shape.")
    if not isinstance(data.get("lossargs"), dict) or not isinstance(
        data.get("penaltyargs"), dict
    ) or not isinstance(data.get("adam"), dict):
        raise SerializationError("Checkpoint metadata has invalid parameters.")
    _verify_checksums(p)
    mlp = MLP(shape)
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
            if "clock" not in loaded.files:
                raise SerializationError("Checkpoint is missing Adam clock.")
            clock = loaded["clock"]
            if clock.shape != () or not np.issubdtype(clock.dtype, np.integer):
                raise SerializationError("Checkpoint has an invalid Adam clock.")
            state: dict[str, object] = {
                "clock": int(clock),
                "mean": {},
                "variance": {},
            }
            for group, params in (("weights", mlp.weights), ("biases", mlp.biases)):
                mean: list[np.ndarray | None] = []
                variance: list[np.ndarray | None] = []
                for i, param in enumerate(params):
                    keymean = f"{group}_{i}_mean"
                    keyvar = f"{group}_{i}_variance"
                    if keymean in loaded.files and keyvar in loaded.files:
                        mean_value = loaded[keymean]
                        variance_value = loaded[keyvar]
                        if (
                            mean_value.shape != param.shape
                            or variance_value.shape != param.shape
                            or not np.all(np.isfinite(mean_value))
                            or not np.all(np.isfinite(variance_value))
                        ):
                            raise SerializationError(
                                f"Invalid Adam buffers for {group}[{i}]."
                            )
                        mean.append(mean_value)
                        variance.append(variance_value)
                    else:
                        mean.append(np.zeros_like(param))
                        variance.append(np.zeros_like(param))
                state["mean"][group] = mean  # type: ignore[index]
                state["variance"][group] = variance  # type: ignore[index]
            try:
                runner.adam.load_state(state)
            except ValueError as exc:
                raise SerializationError("Invalid Adam state.") from exc
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
        raise SerializationError("Checkpoint is missing checksums.json.")
    try:
        checksums = json.loads(checksum_path.read_text())
        if not isinstance(checksums, dict) or not checksums:
            raise SerializationError("Invalid checkpoint checksums.")
        for name, expected in checksums.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise SerializationError("Invalid checkpoint checksums.")
            candidate = path / name
            if candidate.parent != path or not candidate.is_file():
                raise SerializationError(f"Missing checkpoint file {name!r}.")
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
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
