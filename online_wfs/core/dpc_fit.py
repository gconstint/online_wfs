"""Equal-weight affine DPC fits with robust aggregation across valid profiles."""

from typing import Optional, Tuple

import numpy as np

from .arrays import normalize_pixel_size


def _fit_1d_linear(
    data: np.ndarray,
    coords: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_valid_fraction: float = 0.5,
) -> Tuple[Optional[float], Optional[float]]:
    """Fit data = slope * coordinate + offset on valid samples."""
    valid = np.isfinite(data)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if np.count_nonzero(valid) < max(2, len(data) * min_valid_fraction):
        return None, None

    x = coords[valid]
    A = np.column_stack([x, np.ones_like(x)])

    try:
        coeffs, *_ = np.linalg.lstsq(A, data[valid], rcond=None)
        return coeffs[0], coeffs[1]
    except np.linalg.LinAlgError:
        return None, None


def _fit_linear_profiles(data, coords, valid_mask):
    """Batched affine OLS; circle validity is relative to each masked profile.

    Center both coordinates and values before taking moments. Small blocks bound
    temporary memory, including for column views. Rank-deficient designs retain
    the original SVD least-squares behavior rather than dividing by tiny variance.
    """
    slopes, offsets = [], []
    for first in range(0, data.shape[0], 128):
        values = np.asarray(data[first : first + 128], dtype=float)
        valid = np.isfinite(values) & valid_mask[first : first + 128]
        counts = np.count_nonzero(valid, axis=1)
        required = np.maximum(
            2, np.count_nonzero(valid_mask[first : first + 128], axis=1) * 0.5
        )
        eligible = counts >= required
        if not np.any(eligible):
            continue
        safe_counts = np.maximum(counts, 1)
        mean_x = np.sum(np.where(valid, coords, 0.0), axis=1) / safe_counts
        mean_y = np.sum(np.where(valid, values, 0.0), axis=1) / safe_counts
        centered_x = np.where(valid, coords - mean_x[:, None], 0.0)
        centered_y = np.where(valid, values - mean_y[:, None], 0.0)
        variance = np.einsum("ij,ij->i", centered_x, centered_x)
        covariance = np.einsum("ij,ij->i", centered_x, centered_y)

        # Singular-value ratio of [coords, 1], evaluated without subtracting
        # nearly equal eigenvalues. Match lstsq's default rank cutoff.
        norm_x = variance + counts * mean_x**2
        largest = 0.5 * (
            norm_x + counts + np.hypot(norm_x - counts, 2 * counts * mean_x)
        )
        ratio = np.divide(
            np.sqrt(counts * variance),
            largest,
            out=np.zeros_like(largest),
            where=largest > 0,
        )
        stable = eligible & (ratio > 8 * np.finfo(float).eps * np.maximum(counts, 2))
        block_slopes = np.divide(
            covariance, variance, out=np.zeros_like(covariance), where=stable
        )
        block_offsets = mean_y - block_slopes * mean_x
        for index in np.flatnonzero(eligible & ~stable):
            slope, offset = _fit_1d_linear(
                values[index],
                coords,
                valid[index],
                min_valid_fraction=0.0,
            )
            if slope is None:
                eligible[index] = False
            else:
                block_slopes[index], block_offsets[index] = slope, offset
        slopes.extend(block_slopes[eligible].tolist())
        offsets.extend(block_offsets[eligible].tolist())
    return slopes, offsets


def _curvature_radius(
    slope: float,
    wavelength: float,
):
    """Signed curvature radius: 1/R = wavelength * slope / (2*pi)."""
    a = slope / 2.0  # Quadratic coefficient [rad/m²]
    curvature = wavelength * a / np.pi  # Curvature of the supplied DPC [1/m]

    if abs(curvature) > 1e-14:
        R_real = 1.0 / curvature
    else:
        R_real = np.inf

    return R_real


def fit_dpc_background(dpc_x, dpc_y, pixel_size, wavelength, *, valid_mask):
    """Fit X rows and Y columns within the concentric circle; preserve raw DPC.

    Slopes/intercepts are aggregated by median. The coordinate origin follows
    the upstream DPC fit convention (index - (size-1)/2) * sampling.
    """
    dpc_x, dpc_y = np.asarray(dpc_x, float), np.asarray(dpc_y, float)
    if dpc_x.ndim != 2 or dpc_x.shape != dpc_y.shape:
        raise ValueError("DPC arrays must be matching two-dimensional arrays")
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != dpc_x.shape:
        raise ValueError("valid_mask must match DPC shape")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive")
    rows, columns = dpc_x.shape
    py, px = normalize_pixel_size(pixel_size)
    y = (np.arange(rows) - (rows - 1) / 2) * py
    x = (np.arange(columns) - (columns - 1) / 2) * px
    slopes_x, offsets_x = _fit_linear_profiles(dpc_x, x, mask)
    slopes_y, offsets_y = _fit_linear_profiles(dpc_y.T, y, mask.T)
    if not slopes_x or not slopes_y:
        raise ValueError("DPC background fitting requires more valid measurements")
    slope_x, offset_x = np.median(slopes_x), np.median(offsets_x)
    slope_y, offset_y = np.median(slopes_y), np.median(offsets_y)
    fit_x = np.broadcast_to(slope_x * x + offset_x, dpc_x.shape).copy()
    fit_y = np.broadcast_to((slope_y * y + offset_y)[:, None], dpc_y.shape).copy()
    return {
        "fit_params": {
            "slope_x": float(slope_x),
            "slope_y": float(slope_y),
            "a_x": float(slope_x / 2),
            "a_y": float(slope_y / 2),
            "R_x": float(_curvature_radius(slope_x, wavelength)),
            "R_y": float(_curvature_radius(slope_y, wavelength)),
            "x0": float(-offset_x / slope_x) if abs(slope_x) > 1e-14 else 0.0,
            "y0": float(-offset_y / slope_y) if abs(slope_y) > 1e-14 else 0.0,
        },
        "dpc_fit": {"dpc_x_fit": fit_x, "dpc_y_fit": fit_y},
    }
