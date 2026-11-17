"""Bounded-memory OSA/ANSI fitting calculated independently for each phase map."""

from math import factorial

import numpy as np

from ..core.arrays import coordinate_grid
from ..defaults import ZERNIKE_NUM_TERMS, ZERNIKE_TUKEY_ALPHA


def zernike_basis(rho, theta, terms):
    """OSA/ANSI j=0..terms-1 with unit-disk RMS normalization."""
    basis = np.empty((rho.size, terms))
    for j in range(terms):
        n = 0
        while (n + 1) * (n + 2) // 2 <= j:
            n += 1
        m = 2 * j - n * (n + 2)
        radial = np.zeros_like(rho)
        for k in range((n - abs(m)) // 2 + 1):
            coefficient = (
                (-1) ** k
                * factorial(n - k)
                / (
                    factorial(k)
                    * factorial((n + abs(m)) // 2 - k)
                    * factorial((n - abs(m)) // 2 - k)
                )
            )
            radial += coefficient * rho ** (n - 2 * k)
        angular = np.cos(m * theta) if m > 0 else np.sin(abs(m) * theta) if m < 0 else 1
        basis[:, j] = radial * angular * np.sqrt((2 if m else 1) * (n + 1))
    return basis


def fit_zernike_phase(
    phase,
    pixel_size,
    mask,
    aperture_radius_m,
    *,
    terms=ZERNIKE_NUM_TERMS,
    center_m=(0.0, 0.0),
    tukey_alpha=ZERNIKE_TUKEY_ALPHA,
):
    """Fit the current phase on its measured support and return the residual."""
    if isinstance(terms, bool) or int(terms) != terms or not 1 <= terms <= 136:
        raise ValueError("zernike_num_terms must be an integer in [1,136]")
    terms = int(terms)
    if mask.sum() <= terms or not 0 <= tukey_alpha <= 1:
        raise ValueError("insufficient Zernike samples or invalid radial Tukey weight")
    x, y = coordinate_grid(phase.shape, pixel_size)
    rho = np.hypot(x - center_m[0], y - center_m[1]) / aperture_radius_m
    theta = np.arctan2(y - center_m[1], x - center_m[0])
    del x, y
    indices = np.flatnonzero(mask)
    gram, rhs = np.zeros((terms, terms)), np.zeros(terms)

    def blocks():
        for start in range(0, len(indices), 16384):
            idx = indices[start : start + 16384]
            yield idx, zernike_basis(rho.ravel()[idx], theta.ravel()[idx], terms)

    for idx, basis in blocks():
        radial = rho.ravel()[idx]
        weights = np.ones(len(idx))
        if tukey_alpha:
            rim = radial > 1 - tukey_alpha
            weights[rim] = 0.5 * (
                1 + np.cos(np.pi * (radial[rim] - 1 + tukey_alpha) / tukey_alpha)
            )
        gram += basis.T @ (weights[:, None] * basis)
        rhs += basis.T @ (weights * phase.ravel()[idx])
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise ValueError("Zernike fit is rank deficient on measured support")
    coefficients = np.linalg.solve(gram, rhs)
    residual = np.full(phase.shape, np.nan)
    for idx, basis in blocks():
        residual.ravel()[idx] = phase.ravel()[idx] - basis @ coefficients
    return residual
