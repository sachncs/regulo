# Engineering contract

Regulo is a research reproduction with production-minded quality gates. It is
not a general-purpose serving framework. Public APIs are typed, validated, and
backward-compatible within a minor release where practical.

## Architecture

The package is layered as:

```text
public API -> contracts -> validation -> model/loss/penalty/optimizer
           -> training -> metrics/tuning -> checkpoint persistence
```

`TrainingConfig`, `TrainingResult`, and `Checkpoint` are the typed boundaries
between orchestration and numerical kernels. `Loss`, `Penalty`, and `Metric`
are polymorphic extension points; callers should depend on those interfaces,
not on concrete implementation classes.

## Naming and state policy

- Public classes, functions, and attributes use `snake_case` or `PascalCase`.
- A leading underscore marks semi-private implementation state.
- Public state objects are returned as copies and loaded through explicit
  `state()` / `load_state()` contracts.
- Acronyms are treated as words (`xval`, `mlp`), except for established public
  names kept for compatibility.

New mutable caches, optimizer buffers, and implementation details must be
semi-private. A compatibility adapter is required before changing an existing
public name.

## Errors and compatibility

Use `ValidationError` for caller data/configuration and `SerializationError`
for incompatible or corrupt checkpoints. Error messages must identify the
offending field, shape, or file. Schema version changes are explicit; library
version changes alone never silently reinterpret model bytes.

The supported Python range is the range in `pyproject.toml`. Numerical
reproducibility means identical seeds, pinned dependency versions, the same
BLAS family, and tolerance-based equality; bit-for-bit equality across
platforms is not promised.

## Required quality gate

```bash
.venv/bin/ruff check regulo tests
.venv/bin/mypy regulo
.venv/bin/python -m pytest --cov=regulo --cov-fail-under=95 --doctest-modules
(cd site && npm ci && npm run build)
```

Every public API change requires tests, a Google-style docstring update, and a
documentation/example update when behavior changes.
