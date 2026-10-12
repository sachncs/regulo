"""Small reproducible benchmark for regression tracking."""

import time

import numpy as np

from regulo import MLP, Adam, Runner, Square, Void


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 20))
    y = rng.standard_normal((256, 1))
    runner = Runner(
        MLP([20, 32, 1], seed=0), Square(), Void(), Adam(), epochs=5
    )
    start = time.perf_counter()
    runner.fit(x, y, seed=0)
    elapsed = time.perf_counter() - start
    print(f"fit_256x20x32x1_seconds={elapsed:.6f}")


if __name__ == "__main__":
    main()
