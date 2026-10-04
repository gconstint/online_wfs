"""OSA Zernike fitting on the reconstruction circle, with a cached projector."""

import numpy as np
from scipy.linalg import svd
from scipy.special import factorial

from ..core.arrays import circular_domain, normalize_pixel_size
from ..defaults import ZERNIKE_NUM_TERMS, ZERNIKE_TUKEY_ALPHA


def zernike_polynomial(index, rho, theta):
    """Evaluate an OSA/ANSI normalized polynomial by its zero-based index."""
    n = 0
    while (n + 1) * (n + 2) / 2 <= index:
        n += 1
    m = 2 * index - n * (n + 2)
    radial = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        coefficient = (-1) ** k * factorial(n - k)
        coefficient /= (
            factorial(k)
            * factorial((n + abs(m)) // 2 - k)
            * factorial((n - abs(m)) // 2 - k)
        )
        radial += coefficient * rho ** (n - 2 * k)
    angular = np.cos(m * theta) if m > 0 else np.sin(abs(m) * theta) if m < 0 else 1.0
    return (radial * angular) * np.sqrt(2 * (n + 1) if m != 0 else n + 1)


class ZernikeFitter:
    """Reuse one basis/projector; rebuild when physical geometry/support changes."""

    def __init__(self):
        self._geometry_key = None
        self._valid = None

    def fit(self, phase, pixel_size, wavelength, geometry):
        sampling = normalize_pixel_size(pixel_size)
        center, radius = tuple(geometry["center_m"]), float(geometry["radius_m"])
        key = (phase.shape, sampling, center, radius)
        if key != self._geometry_key:
            mask, _ = circular_domain(
                phase.shape, sampling, center_m=center, radius_m=radius
            )
            rows, columns = phase.shape
            y = (np.arange(rows) - rows / 2) * sampling[0] - center[1]
            x = (np.arange(columns) - columns / 2) * sampling[1] - center[0]
            rho = np.hypot(y[:, None], x[None, :]) / radius
            theta = np.arctan2(y[:, None], x[None, :])
            self._basis = np.stack(
                [
                    np.where(mask, zernike_polynomial(j, rho, theta), 0.0)
                    for j in range(ZERNIKE_NUM_TERMS)
                ],
                axis=-1,
            )
            weights = np.ones(phase.shape)
            alpha = ZERNIKE_TUKEY_ALPHA
            edge = (rho >= 1 - alpha) & (rho <= 1)
            weights[edge] = 0.5 * (
                1 + np.cos(np.pi * (rho[edge] - (1 - alpha)) / alpha)
            )
            self._sqrt_weights = np.sqrt(np.where(mask, weights, 0.0))
            self._mask, self._geometry_key, self._valid = mask, key, None
        valid = self._mask & np.isfinite(phase)
        if valid.sum() < ZERNIKE_NUM_TERMS:
            raise ValueError(
                "not enough measured pixels for the default 36 Zernike terms"
            )
        if self._valid is None or not np.array_equal(valid, self._valid):
            self._matrix = self._basis[valid]
            weighted = self._matrix * self._sqrt_weights[valid, None]
            u, singular, vh = svd(weighted, full_matrices=False, check_finite=False)
            cutoff = np.finfo(float).eps * max(weighted.shape) * singular[0]
            inverse = np.divide(
                1.0, singular, out=np.zeros_like(singular), where=singular > cutoff
            )
            self._projector = (vh.T * inverse) @ u.T
            self._valid = valid.copy()
        coefficients = np.einsum(
            "ij,j->i", self._projector, phase[valid] * self._sqrt_weights[valid]
        )
        fitted = np.full(phase.shape, np.nan)
        fitted[valid] = np.einsum("ij,j->i", self._matrix, coefficients)
        residual = np.where(valid, phase - fitted, np.nan)
        rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
        return {
            "coefficients": coefficients,
            "fitted_phase": fitted,
            "residual": residual,
            "rms_error": rms,
            "rms_error_nm": rms * wavelength / (2 * np.pi) * 1e9,
        }
