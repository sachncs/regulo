# Fidelity Report

This document compares every major component of the paper against its implementation, flagging exact matches, assumptions, and known deviations.

---

## Regularizers

| Component | Paper Definition | Implementation | Status |
|---|---|---|---|
| Ridge | `λ ‖W‖_F^2` | `Ridge(lam)` | **Exact** |
| Lasso | `γ ‖W‖_1` | `Lasso(gamma)` | **Exact** |
| Elastic Net | `α γ ‖W‖_1 + (1-α)/2 ‖W‖_F^2` | `ElasticNet(alpha, gamma)` | **Exact** |
| Covridge | `λ_1 ‖C_{δ,n}^{1/2} W‖_F^2 + λ_2 ‖W‖_F^2` | `Covridge(lambda1, lambda2, gram)` | **Exact** |
| Sparridge | `λ_1 ‖C_{δ,n}^{1/2} W‖_F^2 + γ ‖W‖_1` | `Sparridge(lambda1, gamma, gram)` | **Exact** |

**Assumptions:**
- The paper defines a single `C_{δ,n}` based on the input matrix `H`. In a multi-layer network, this matrix only matches the first-layer weight shape. We apply Covridge/Sparridge **only to the first layer**, documented as an assumption.

---

## Network Architecture

| Component | Paper | Implementation | Status |
|---|---|---|---|
| Regression hidden layers | 64 and 32 units | `MLP(shape=[p, 64, 32, 1])` | **Exact** |
| Classification hidden layers | 8 and 4 units | `MLP(shape=[p, 8, 4, k])` | **Exact** |
| Activation | ReLU | `np.maximum(z, 0.0)` | **Exact** |
| Output (regression) | Linear | Linear | **Exact** |
| Output (classification) | Softmax | Softmax inside loss | **Exact** |
| Initialization | Unspecified | Xavier uniform | **Assumed** |

---

## Optimization

| Component | Paper | Implementation | Status |
|---|---|---|---|
| Optimizer | Adam (default settings) | `Adam(beta1=0.9, beta2=0.999, epsilon=1e-8)` | **Exact** |
| Learning rate | Unspecified | `lr=1e-3` | **Assumed** |
| Epochs | 500 | `epochs=500` | **Exact** |
| Batch size (regression) | 32 | `batch=32` | **Exact** |
| Batch size (classification) | 16 | `batch=16` | **Exact** |
| Early stopping | Patience 10 (classification) | `earlystop=True, patience=10` | **Exact** |

---

## Loss Functions

| Component | Paper | Implementation | Status |
|---|---|---|---|
| Regression | MSE | `regulo.loss.Square` | **Exact** |
| Classification | Cross-entropy | `regulo.loss.Softmax` (softmax + NLL) | **Exact** |

---

## Data

| Component | Paper | Implementation | Status |
|---|---|---|---|
| DGP1 | (200, 20, 10) | `regulo.data.synth(n=200, p=20, k=10, ...)` | **Exact** |
| DGP2 | (1000, 200, 100) | `regulo.data.synth(n=1000, p=200, k=100, ...)` | **Exact** |
| DGP3 | (500, 2000, 100) | `regulo.data.synth(n=500, p=2000, k=100, ...)` | **Exact** |
| Correlation | ρ ∈ {0.25, 0.75} | Parameter `rho` | **Exact** |
| Noise | σ ∈ {0.10, 2.00} | Parameter `noise` | **Exact** |
| Linear signal | `y = Xθ + ε` | `regulo.data.synth(nonlinear=False)` | **Exact** |
| Nonlinear signal | `y = Σ θ_j sin(x_j) + ε` | `regulo.data.synth(nonlinear=True)` | **Exact** |
| Train/test split | 75/25 | `int(0.75 * n)` via `split` in `demo/run_simulation.py` | **Exact** |
| Standardization | Training stats only | `regulo.tune.Scaler` inside CV | **Exact** |

---

## Cross-Validation

| Component | Paper | Implementation | Status |
|---|---|---|---|
| k-fold | 5-fold | `regulo.tune.kfold(n, folds=5, seed=...)` | **Exact** |
| Simulation grid | {0.001, 0.01, 0.1, 0.5, 0.9} | `GRID` in `demo/run_simulation.py` | **Exact** |
| Grid evaluation | All combinations | Nested loops over grid | **Exact** |

---

## Evaluation Metrics

| Metric | Paper | Implementation | Status |
|---|---|---|---|
| MSE | Reported | `regulo.score.Mse` | **Exact** |
| MAE | Reported | `regulo.score.Mae` | **Exact** |
| RMSE | Reported | `regulo.score.Rmse` | **Exact** |
| R² | Reported | `regulo.score.R2` | **Exact** |
| Balanced accuracy | Reported | `regulo.score.Balanced` | **Exact** |

---

## Real-Data Experiments

| Experiment | Paper | Implementation | Status |
|---|---|---|---|
| UCI Energy | 768 samples, 8 features | Removed in v0.1.x; see `docs/limits.md` §2 | **Removed** |
| GSE9476 | 64 samples, 22,000 genes, ANOVA→2000 | Removed in v0.1.x; see `docs/limits.md` §2 | **Removed** |

**Note:** Earlier versions of this package included real-data loaders
backed by `fetch_openml`.  These were dropped when the library was
rewritten in pure NumPy + SciPy with no scikit-learn dependency; see
`docs/limits.md` §2 for the rationale.  Users supply their own arrays
directly via `regulo.tune.Scaler` for standardization.

---

## Known Deviations

1. **Initialization:** Paper does not specify weight initialization. We use Xavier uniform, a standard choice.
2. **δ value:** Paper defines `C_{δ,n} = C_n + δ I` but does not specify `δ`. We use `1e-4`.
3. **Learning rate:** Paper says "Adam default settings" but does not state the learning rate. We use `1e-3`.
4. **Layer-wise Covridge/Sparridge:** The paper's `C_{δ,n}` is computed from the input dimension, so these penalties are applied only to the first layer. This is documented as an assumption.
5. **Theorems 5.1 and 5.2:** These describe asymptotic statistical properties and are not implemented. Only the empirical training algorithm is reproduced.
