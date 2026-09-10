"""Model serialization via NPZ + JSON.

Save and load :class:`regulo.fit.Runner` state to a directory of
NPZ and JSON files.  No ``pickle`` is used; the on-disk format
cannot execute arbitrary code on load.  The schema is stable and
versioned against :data:`regulo.__version__`.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from regulo import __version__
from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Loss
from regulo.net import MLP
from regulo.penalty import Penalty, REGISTRY

__all__ = ["save", "load", "snapshot", "meta"]


def meta(runner: Runner) -> Dict:
    """Build the metadata dictionary describing *runner*."""
    

    return {
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
    """Persist a Runner to *path*.

    Creates a directory *path* containing:
      * ``meta.json`` -- architecture and hyperparameters
      * ``weights.npz`` -- one ``w{i}`` array per weight matrix
      * ``biases.npz`` -- one ``b{i}`` array per bias vector
      * ``adam.npz`` -- ``mean`` / ``variance`` / ``clock`` per group
      * ``gram.npy`` -- the geometry-aware Gram matrix (when the
        penalty is Covridge or Sparridge)
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "meta.json", "w") as f:
        json.dump(meta(runner), f, indent=2, default=serialize)
    np.savez(
        p / "weights.npz",
        **{f"w{i}": w for i, w in enumerate(runner.mlp.weights)},
    )
    np.savez(
        p / "biases.npz",
        **{f"b{i}": b for i, b in enumerate(runner.mlp.biases)},
    )
    data = {}
    for group in runner.adam.mean:
        for i, buf in enumerate(runner.adam.mean[group]):
            data[f"{group}_{i}_mean"] = (
                np.zeros_like(runner.mlp.weights[i])
                if buf is None
                else buf
            )
        for i, buf in enumerate(runner.adam.variance[group]):
            data[f"{group}_{i}_variance"] = (
                np.zeros_like(runner.mlp.weights[i])
                if buf is None
                else buf
            )
    data["clock"] = np.array(runner.adam.clock)
    np.savez(p / "adam.npz", **data)

    geom = getattr(runner.penalty, "csqrt", None)
    if geom is not None:
        gram = geom @ geom
        np.save(p / "gram.npy", gram)


def snapshot(runner: Runner) -> Dict:
    """Alias for :func:`meta`.  Return the metadata dictionary."""
    return meta(runner)


def load(path: str) -> Runner:
    """Reconstruct a Runner from *path*."""
    p = Path(path)
    with open(p / "meta.json") as f:
        data = json.load(f)
    major = data["version"].split(".")[0]
    lib = __version__.split(".")[0]
    if major != lib:
        raise ValueError(
            f"version mismatch: on-disk {data['version']!r}, "
            f"library {__version__!r}."
        )
    mlp = MLP(data["shape"])
    cls = losslookup(data["loss"])
    loss = cls(**data["lossargs"])
    penalty = penaltylookup(data, p)
    adam = Adam(**data["adam"])

    weights = np.load(p / "weights.npz")
    biases = np.load(p / "biases.npz")
    for i, w in enumerate(mlp.weights):
        w[...] = weights[f"w{i}"]
    for i, b in enumerate(mlp.biases):
        b[...] = biases[f"b{i}"]

    runner = Runner(mlp, loss, penalty, adam)

    # Restore Adam state if compatible.
    adampath = p / "adam.npz"
    if adampath.exists():
        loaded = np.load(adampath)
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
                        runner.mlp.weights[i] if group == "weights" else runner.mlp.biases[i]
                    )
                    runner.adam.variance[group][i] = np.zeros_like(
                        runner.mlp.weights[i] if group == "weights" else runner.mlp.biases[i]
                    )
    return runner


def lossargs(loss: Loss) -> Dict:
    return {}


def penaltyargs(penalty: Penalty) -> Dict:
    out = {}
    for name in penalty.hp:
        if name == "gram":
            continue
        if hasattr(penalty, name):
            out[name] = getattr(penalty, name)
    return out


def losslookup(name: str):
    from regulo.loss import Softmax, Square

    table = {"square": Square, "softmax": Softmax}
    if name not in table:
        raise ValueError(f"Unknown loss in meta: {name!r}")
    return table[name]


def penaltylookup(data: Dict, path: Path) -> Penalty:
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
        hp["gram"] = np.load(gram_path)
    cls = REGISTRY[name]
    return cls(**hp)


def serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
