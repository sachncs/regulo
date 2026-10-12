"""Property-style invariants for the numerical building blocks."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from regulo.net import MLP, xavier


@given(
    fanin=st.integers(min_value=1, max_value=8),
    fanout=st.integers(min_value=1, max_value=8),
)
def test_xavier_property(fanin: int, fanout: int):
    weights = xavier(fanin, fanout, np.random.default_rng(0))
    assert weights.shape == (fanin, fanout)
    assert np.all(np.isfinite(weights))


@given(width=st.integers(min_value=1, max_value=8))
def test_mlp_shape_property(width: int):
    model = MLP([2, width, 1], seed=0)
    output = model(np.zeros((3, 2)))
    assert output.shape == (3, 1)
    assert all(np.all(np.isfinite(value)) for value in output)
