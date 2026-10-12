"""Smoke tests for the documented reproduction helpers."""

import numpy as np

from demo.run_simulation import buildgrid, split


def test_demo_grid_and_split_are_deterministic():
    assert buildgrid("ridge") == [{"lam": 0.01}, {"lam": 0.1}, {"lam": 0.5}]
    x = np.arange(20).reshape(10, 2)
    y = np.arange(10).reshape(10, 1)
    first = split(x, y, 0.2, seed=4)
    second = split(x, y, 0.2, seed=4)
    for left, right in zip(first, second, strict=False):
        np.testing.assert_array_equal(left, right)
