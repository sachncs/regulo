"""Synthetic data generators matching the paper's DGP designs.

Provides the :func:`synth` generator with the equi-correlation
covariance structure used in the paper's experiments.

Common mathematical model
-------------------------
.. math::

    X_{\\text{info}} &\\sim \\mathcal{N}(0, \\Sigma_k) \\quad
        \\text{where } \\Sigma_k = (1-\\rho)I_k + \\rho \\mathbf{1}\\mathbf{1}^T
    \\\\
    X_{\\text{noise}} &\\sim \\mathcal{N}(0, I_{p-k})
    \\\\
    \\theta &\\sim \\mathcal{N}(0, \\tau I_k)
    \\\\
    y &= X_{\\text{info}} \\theta + \\varepsilon
        \\quad\\text{(or } \\sin(X_{\\text{info}}) \\theta + \\varepsilon\\text{)}

where ``epsilon ~ N(0, sigma_noise^2 I)``.

Real-data experiments are out of scope: bring your own arrays and
pass them to :class:`regulo.tune.Scaler` and :class:`Runner` for
standardization and training.
"""

from typing import Optional, Tuple

import numpy as np

__all__ = ["equicorr", "synth"]


def equicorr(k: int, rho: float) -> np.ndarray:
    """Build a ``k x k`` equi-correlation covariance matrix.

    Diagonal entries are ``1``, off-diagonal entries are ``rho``.
    The matrix has the closed form
    ``Sigma_k = (1 - rho) I_k + rho * 1_k 1_k^T`` and is positive
    definite exactly when ``rho in (-1/(k-1), 1)``.

    Args:
        k: Dimension of the matrix.  Must be positive.
        rho: Off-diagonal correlation value.

    Returns:
        Covariance matrix of shape ``(k, k)``.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > 1 and not (-1.0 / (k - 1) < rho < 1.0):
        raise ValueError(
            f"rho must be in (-1/(k-1), 1) for k={k}; got {rho}."
        )
    sigma = np.full((k, k), rho)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def synth(
    n: int,
    p: int,
    k: int,
    rho: float,
    noise: float,
    tau: float = 1.0,
    nonlinear: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data according to the paper's DGP.

    The first ``k`` features are drawn from a multivariate normal
    with equi-correlation ``rho``; the remaining ``p - k`` features
    are i.i.d. standard normal.  True coefficients are drawn from
    ``N(0, tau)`` and the response is either linear or sinusoidal.

    Args:
        n: Number of samples.  Must be positive.
        p: Total number of features.  Must be positive.
        k: Number of informative (correlated) features.  Must
            satisfy ``0 <= k <= p``.
        rho: Pairwise correlation among informative features.
        noise: Standard deviation of additive Gaussian noise.
        tau: Standard deviation of the true coefficient vector.
        nonlinear: If ``True``, use ``sin(X) @ theta`` instead of
            ``X @ theta``.
        seed: Seed for the NumPy default RNG.

    Returns:
        ``(X, y)`` with shapes ``(n, p)`` and ``(n, 1)``.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if p <= 0:
        raise ValueError("p must be positive.")
    if not (0 <= k <= p):
        raise ValueError(f"k must be in [0, p]; got k={k}, p={p}.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")
    if tau < 0:
        raise ValueError("tau must be non-negative.")
    if k > 1 and not (-1.0 / (k - 1) < rho < 1.0):
        raise ValueError(
            f"rho must be in (-1/(k-1), 1) for k={k}; got {rho}."
        )

    rng = np.random.default_rng(seed)
    if k > 0:
        cov = equicorr(k, rho)
        info = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    else:
        info = np.empty((n, 0))
    extra = rng.standard_normal(size=(n, p - k))
    x = np.hstack([info, extra])
    theta = rng.normal(0.0, tau, size=k) if k > 0 else np.zeros(0)
    if k == 0:
        y = np.zeros(n)
    elif nonlinear:
        y = np.sum(theta * np.sin(info), axis=1)
    else:
        y = np.ravel(info @ theta)
    y = y + rng.normal(0.0, noise, size=n)
    return x, y.reshape(-1, 1)
