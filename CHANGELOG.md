# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive docstrings across all 9 `anbr/` modules with module,
  class, and method docstrings including Args/Returns/Raises sections,
  algorithm explanations, edge-case documentation, paper equation
  mappings, and numerical stability notes (`7811fb7` — 2026-07-11)
- Rebuilt README with centered header, badge block, Quick Start
  (CLI + Python API), Code Style, Commit Conventions, and Tech Stack
  subsections (`ecb68ad` — 2026-07-11)
- Refined docstrings across `anbr/cv.py`, `anbr/data.py`, `anbr/losses.py`,
  `anbr/metrics.py`, `anbr/network.py`, `anbr/optimizer.py`,
  `anbr/trainer.py`, and `tests/__init__.py` -- documented the `1/N`
  MSE averaging convention, the softmax numerical-stability
  subtraction and log-floor, the lazy Adam moment-buffer
  initialisation contract, the Trainer single-run lifecycle, the
  closed-form equi-correlation covariance, the per-class-recall
  balanced-accuracy formula, and the cross-validation training
  budget (`0644805` — 2026-07-12)
- New README Architecture section with gradient-flow diagram, five
  mathematical-guarantee statements, and Lifecycle note; introduced
  Configuration tables (Network Architecture, Regularizers, Optimizer)
  and a dedicated Tech Stack table; corrected Code Style line-length
  claim (80, matching `pyproject.toml`) and added a `grid_search_cv`
  Quick Start example (`ddacc2a` — 2026-07-12)

### Changed

- Removed unused imports: `Tuple` from network, `Any`/`Tuple`/`StandardScaler`
  from trainer, `Callable` from cv, `StandardScaler` from run_real_data,
  `pytest` from test_network, `build_regularizer` from test_integration
  (`468d98b` — 2026-07-11)
- Applied black formatting to test files for style consistency
  (`2b8316d` — 2026-07-11)
- Updated CHANGELOG with all commit IDs and dates across v0.1.0,
  v0.1.1, and Unreleased (`af255b2` — 2026-07-11)
- `CONTRIBUTING.md` -- switched the Docstring Format section from
  NumPy-style to Google-style (matching the convention used in
  `anbr/`) and corrected the PR-target branch from `main` to `master`
- `SECURITY.md` -- replaced `[INSERT SECURITY EMAIL]` placeholders
  with the project's actual contact address

### Fixed

- Corrected GitHub username from `sachn-cs` to `sachncs` across README,
  CHANGELOG, CONTRIBUTING, pyproject.toml, FUNDING.yml, and
  getting-started.md (`59ab064` — 2026-07-11)
- Updated LICENSE copyright holder to `Sachin` (`c73e254` — 2026-07-11)
- Bumped mypy `python_version` from `"3.10"` to `"3.12"` in
  pyproject.toml for numpy stub compatibility (`c5712a1` — 2026-07-11)
- Ensured 1D array type consistency in `data.py` using `np.ravel()`
  for Python 3.10 mypy compatibility (`8a1836b` — 2026-07-11)
- Removed stale `.mypy.ini` that overrode pyproject.toml
  `python_version` setting, causing numpy PEP 695 `type` syntax
  errors on CI (`221e1d9` — 2026-07-11)
- Cast `np.linspace` result to native `float` in `run_real_data.py`
  for mypy `list[float]` compatibility (`813343d` — 2026-07-11)
- Removed misleading `Raises: ValueError (implicit via broadcasting)`
  on `MSELoss.forward` and replaced it with an honest Notes block
  describing NumPy broadcasting behaviour (`0644805` — 2026-07-12)
- Corrected the outdated line-length claim in the README from "88
  (black default)" to the actual `pyproject.toml` setting of 80
  (`ddacc2a` — 2026-07-12)
- Resolved ruff `F401` unused-import warning on `anbr.data.make_dgp3`
  in `demo/run_simulation.py` (`672bb76` — 2026-07-12)
- Made `regulo.__version__` single source of truth between pyproject.toml and regulo 
  package

### Removed

- Removed `.env.example` — project requires no environment variables;
  all hyperparameters are passed programmatically or via CLI arguments

## [0.1.1] - 2026-06-19

### Added

- Open source community files: CODE_OF_CONDUCT.md, SECURITY.md
  (`5fd67e7`..`e519252`)
- Expanded CONTRIBUTING.md with detailed guidelines (branching, commits, PRs)
  (`5fd67e7`..`e519252`)
- GitHub issue templates (bug report, feature request) and pull request
  template (`5fd67e7`..`e519252`)
- Dependabot configuration for automated GitHub Actions dependency updates
  (`5fd67e7`..`e519252`)
- EditorConfig for consistent formatting across editors
  (`5fd67e7`..`e519252`)
- GitAttributes for line-ending normalization
  (`5fd67e7`..`e519252`)
- Getting started guide (`docs/getting-started.md`) and FAQ (`docs/faq.md`)
  (`5fd67e7`..`e519252`)
- CHANGELOG.md following Keep a Changelog format
  (`5fd67e7`..`e519252`)
- Comprehensive README with badges, usage examples, and project structure
  (`5fd67e7`..`e519252`)
- FUNDING.yml for sponsorship configuration
  (`5fd67e7`..`e519252`)

### Changed

- Updated pyproject.toml with full project metadata
  (`5fd67e7`..`e519252`)
- Enhanced CI workflow with build verification step
  (`5fd67e7`..`e519252`)
- Improved .gitignore with additional patterns
  (`5fd67e7`..`e519252`)

### CI

- Bump `actions/checkout` from v4 to v7
  (`8929311` — 2026-06-19, merged `0e07d96` — 2026-07-07)
- Bump `actions/setup-python` from v5 to v6
  (`04336c8` — 2026-06-19, merged `46484b8` — 2026-07-07)

## [0.1.0] - 2026-05-06

### Added

- All 6 regularizers: NoReg, Ridge, Lasso, Elastic Net, Covridge, Sparridge
- Feedforward ReLU network with manual forward/backward propagation
- Adam optimizer implemented from scratch in NumPy
- MSE and softmax cross-entropy losses with analytical derivatives
- 3 DGP generators matching paper's simulation designs (DGP1/2/3)
- UCI Energy Efficiency data loader
- GSE9476 leukemia data loader
- k-fold cross-validation with hyperparameter grid search
- Evaluation metrics: MSE, MAE, RMSE, R², balanced accuracy
- Training loop with mini-batching and early stopping
- Demo scripts for simulations (`run_simulation.py`) and real data
  (`run_real_data.py`)
- 81 unit and integration tests (all passing)
- Full documentation: math foundations, architecture, API reference,
  usage guide
- Paper-to-code fidelity report
- GitHub Actions CI with multi-Python-version matrix testing (3.10–3.13)
- Code formatting with black and isort
- Type checking with mypy

[Unreleased]: https://github.com/sachncs/adaptive-norm-based-regularization/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sachncs/adaptive-norm-based-regularization/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sachncs/adaptive-norm-based-regularization/releases/tag/v0.1.0
