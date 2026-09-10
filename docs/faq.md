# Frequently Asked Questions

## General

### What is this project?

This is a faithful end-to-end pure Python reproduction of the arXiv paper
"Adaptive Norm-Based Regularization for Neural Networks" by Muhammad Qasim
and Farrukh Javed (Lund University). It implements all components from scratch
using NumPy.

### Why use pure NumPy instead of PyTorch/TensorFlow?

The goal is faithful reproduction and educational clarity. Using NumPy allows
each component (network, optimizer, penalty, loss) to be implemented from
scratch, making the mathematics transparent and auditable.

### Can I use this in production?

This project is primarily a research reproduction. While the code is well-tested,
it is optimized for clarity rather than performance. For production use, consider
implementing the penalties in your preferred deep learning framework.

## Installation

### What Python version do I need?

Python 3.10 or later is required.

### Do I need a GPU?

No. This is a pure NumPy implementation that runs on CPU only.

### How do I install development dependencies?

```bash
pip install -e ".[dev]"
```

## Usage

### How do I compare all penalties?

Use the demo script:

```bash
python demo/run_simulation.py      # Synthetic data
```

### How do I add a custom penalty?

Implement the :class:`Penalty` interface:

```python
from regulo.penalty import Penalty
import numpy as np

class MyPenalty(Penalty):
    name = "mine"
    hp = ("weight",)

    def __init__(self, weight: float):
        self.weight = weight

    def value(self, w, layer):
        return self.weight * float(np.sum(w ** 2))

    def grad(self, w, layer):
        return 2.0 * self.weight * w
```

Then add it to :data:`regulo.penalty.REGISTRY`:

```python
from regulo.penalty import REGISTRY
REGISTRY["mine"] = MyPenalty
```

### How do I change the network architecture?

Pass a different ``shape`` argument to :class:`MLP`:

```python
from regulo.net import MLP

net = MLP([10, 128, 64, 32, 1])
```

### What are Covridge and Sparridge?

- **Covridge**: A covariance-weighted ridge penalty that adapts shrinkage to
  the empirical geometry of the inputs
- **Sparridge**: Similar to Covridge but with an L1 sparsity term instead of
  L2

Both outperform standard ridge, lasso, and elastic net in correlated or
high-dimensional settings.

## Testing

### How do I run all tests?

```bash
pytest tests/ -v
```

### How do I run a specific test?

```bash
pytest tests/test_penalty.py -v
pytest tests/test_net.py -k backward_shape -v
```

## Development

### How do I format my code?

```bash
ruff format regulo tests demo
ruff check regulo tests demo
```

### How do I check types?

```bash
mypy regulo
```

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

## Troubleshooting

### Tests fail with import errors

Ensure you've installed in editable mode:

```bash
pip install -e ".[dev]"
```

### Slow performance

NumPy performance depends on BLAS configuration. Check with:

```bash
python -c "import numpy; numpy.show_config()"
```
