"""Replicate simulation experiments (Tables 1-3) on a reduced scale.

The paper uses 100 Monte Carlo replications.  By default this script
runs 5 replications for speed; pass ``--full`` for 100.

Run from the repository root after installing regulo::

    pip install -e .
    python demo/run_simulation.py --seed 0
"""

import argparse
from typing import Dict, List

import numpy as np

from regulo.adam import Adam
from regulo.fit import Runner
from regulo.loss import Square
from regulo.net import MLP
from regulo.penalty import ElasticNet, Lasso, Ridge, Sparridge, Void
from regulo.score import Mse
from regulo.tune import Scaler, resolve, search

METHODS = ["none", "ridge", "lasso", "elastic_net", "covridge", "sparridge"]

# A reduced grid for the demo; the full paper grid is {0.001, 0.01,
# 0.1, 0.5, 0.9} (25 grid points per geometry-aware method).
GRID = [0.01, 0.1, 0.5]


def buildgrid(method: str) -> List[Dict[str, float]]:
    """Return the hyperparameter search grid for *method*."""
    if method == "none":
        return [{}]
    if method == "ridge":
        return [{"lam": v} for v in GRID]
    if method == "lasso":
        return [{"gamma": v} for v in GRID]
    if method == "elastic_net":
        return [{"alpha": a, "gamma": g} for a in [0.5] for g in GRID]
    if method == "covridge":
        return [
            {"lambda1": a, "lambda2": b} for a in GRID for b in GRID
        ]
    if method == "sparridge":
        return [{"lambda1": a, "gamma": g} for a in GRID for g in GRID]
    return []


def evaluate(
    method: str,
    xtrain: np.ndarray,
    ytrain: np.ndarray,
    xtest: np.ndarray,
    ytest: np.ndarray,
    shape: List[int],
    epochs: int,
    seed: int,
) -> Dict[str, float]:
    """Cross-validate, retrain, and evaluate a single method."""
    grid = buildgrid(method)
    if len(grid) == 1 and grid[0] == {}:
        best: Dict[str, float] = {}
    else:
        best, _ = search(
            xtrain,
            ytrain,
            shape,
            method,
            grid,
            loss=Square(),
            folds=3,
            epochs=max(50, epochs // 2),
            lr=1e-3,
            seed=seed,
        )

    scaler = Scaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)

    penalty = resolve(method, best, xtrain)
    mlp = MLP(shape, seed=seed)
    adam = Adam(lr=1e-3)
    runner = Runner(
        mlp,
        Square(),
        penalty,
        adam,
        batch=32,
        epochs=epochs,
    )
    runner.fit(xtrain, ytrain, seed=seed)
    preds = runner.predict(xtest)
    return {
        "mse": Mse()(ytest, preds),
    }


def split(
    x: np.ndarray, y: np.ndarray, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random 75/25 (or other) train/test split using numpy."""
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int((1.0 - frac) * n)
    return x[perm[:cut]], x[perm[cut:]], y[perm[:cut]], y[perm[cut:]]


def execute(
    factory,
    name: str,
    shape: List[int],
    reps: int,
    rhos: List[float],
    noises: List[float],
    seed: int,
) -> None:
    """Run all methods on a DGP across multiple replications."""
    for rho in rhos:
        for noise in noises:
            print(f"\n{name} -- rho={rho}, noise={noise}")
            results: Dict[str, List[float]] = {m: [] for m in METHODS}
            for rep in range(reps):
                x, y = factory(rho, noise, rep)
                xtrain, xtest, ytrain, ytest = split(
                    x, y, frac=0.25, seed=seed + rep
                )
                for method in METHODS:
                    res = evaluate(
                        method,
                        xtrain,
                        ytrain,
                        xtest,
                        ytest,
                        shape,
                        epochs=100,
                        seed=seed + rep,
                    )
                    results[method].append(res["mse"])
            print(f"{'Method':<15} {'MSE mean':<12} {'MSE std':<12}")
            for method in METHODS:
                arr = np.array(results[method])
                print(
                    f"{method:<15} {np.mean(arr):<12.4f} {np.std(arr):<12.4f}"
                )


def linear(rho: float, noise: float, rep: int) -> tuple[np.ndarray, np.ndarray]:
    """DGP1 wrapper for the linear signal."""
    from regulo.data import synth

    return synth(
        n=200, p=20, k=10,
        rho=rho, noise=noise,
        nonlinear=False, seed=rep,
    )


def nonlinear(rho: float, noise: float, rep: int) -> tuple[np.ndarray, np.ndarray]:
    """DGP1 wrapper for the nonlinear (sinusoidal) signal."""
    from regulo.data import synth

    return synth(
        n=200, p=20, k=10,
        rho=rho, noise=noise,
        nonlinear=True, seed=rep,
    )


def main() -> None:
    """Entry point: parse CLI args and run DGP experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true", help="Run 100 replications (slow)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)."
    )
    args = parser.parse_args()
    reps = 100 if args.full else 5
    seed = args.seed

    print("=== Simulation Reproduction ===")
    print(f"Replications: {reps}")
    print(f"Seed: {seed}")

    execute(
        linear,
        "DGP1 (linear)",
        [20, 64, 32, 1],
        reps,
        [0.25, 0.75],
        [0.10, 2.00],
        seed,
    )
    execute(
        nonlinear,
        "DGP1 (nonlinear)",
        [20, 64, 32, 1],
        reps,
        [0.25, 0.75],
        [0.10, 2.00],
        seed,
    )


if __name__ == "__main__":
    main()
