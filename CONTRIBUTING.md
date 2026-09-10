# Contributing to Adaptive Norm-Based Regularization

Thank you for your interest in contributing! This document provides guidelines
and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Running Tests](#running-tests)
- [Documentation](#documentation)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
Please read it before contributing.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/adaptive-norm-based-regularization.git
   cd adaptive-norm-based-regularization
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/sachncs/adaptive-norm-based-regularization.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install code quality tools
pip install black isort mypy
```

## Branch Naming

Use descriptive branch names with a type prefix:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New features | `feat/custom-regularizer` |
| `fix/` | Bug fixes | `fix/gradient-calculation` |
| `docs/` | Documentation | `docs/update-api-reference` |
| `refactor/` | Code refactoring | `refactor/optimizer-state` |
| `test/` | Adding or updating tests | `test/covridge-edge-cases` |
| `chore/` | Maintenance tasks | `chore/update-dependencies` |

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, no logic change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Build process, CI/CD, or auxiliary tool changes |

### Examples

```bash
git commit -m "feat(regularizers): add adaptive learning rate support"
git commit -m "fix(network): correct gradient shape in backward pass"
git commit -m "docs(api): add examples for Covridge class"
git commit -m "test(optimizer): add convergence tests for Adam"
git commit -m "chore(ci): add Python 3.13 to test matrix"
```

### Scope

The optional scope should be the module affected:

- `regularizers`, `network`, `optimizer`, `losses`, `data`, `metrics`, `trainer`, `cv`

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass**:
   ```bash
   pytest tests/ -v
   ```

2. **Format your code**:
   ```bash
   black .
   isort .
   ```

3. **Run type checking**:
   ```bash
   mypy anbr tests demo
   ```

4. **Update documentation** if your change affects the API or user-facing behavior

### Submitting a PR

1. Push your branch to your fork
2. Open a Pull Request against `master`
3. Fill out the PR template completely
4. Link any related issues
5. Ensure CI passes

### PR Guidelines

- Keep PRs focused — one feature or fix per PR
- Write clear, descriptive titles
- Include before/after examples for behavioral changes
- Add tests for new functionality
- Update documentation for API changes
- Maintain or improve test coverage

### Review Process

- A maintainer will review your PR within a reasonable timeframe
- Address all review comments
- Once approved, a maintainer will merge your PR

## Coding Standards

### Code Style

- Follow **PEP 8** with an **80-character line limit**
- Use **black** for automatic formatting
- Use **isort** for import sorting (profile: "black", line_length: 80)
- Use **type hints** on all public APIs
- Write **docstrings** for all public classes and functions

### Docstring Format

Use Google-style docstrings with `Args`, `Returns`, and `Raises`
sections -- this matches the convention used throughout `anbr/`:

```python
def regularize(self, weights: np.ndarray) -> float:
    """Compute the regularization penalty.

    Args:
        weights: Weight matrix of shape ``(n_features, n_outputs)``.

    Returns:
        The scalar regularization penalty as a ``float``.
    """
```

A short summary line (one sentence, imperative) is followed by any
necessary elaboration, then by the structured sections.  Where a
function has non-obvious behaviour, add an inline example or a
``Notes:`` block.  The ``why`` of an implementation belongs in the
docstring or as an inline comment; the ``what`` is left to the code
itself.

### Type Hints

```python
from typing import Optional

def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 500,
    batch_size: int = 32,
    regularizer: Optional[Regularizer] = None,
) -> dict:
    """Train the network."""
```

### What to Avoid

- Do not introduce placeholders, TODOs, or stubs
- Do not add dependencies without discussion in an issue
- Do not reformat unrelated code
- Do not remove features unless they are clearly obsolete

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_regularizers.py -v

# Run with coverage
pytest tests/ -v --tb=short

# Run integration tests only
pytest tests/test_integration.py -v
```

## Documentation

- Update `docs/api.md` for API changes
- Update `docs/usage.md` for usage changes
- Update `docs/fidelity.md` if affecting paper-to-code fidelity
- Add examples to docstrings for non-obvious functions
- Keep the README current with project capabilities

## Questions?

Open an issue with the `question` label if you have questions about contributing.
