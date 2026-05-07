"""
DPC (Differential Phase Contrast) preprocessing module.

Utilities for DPC fitting and relative-reference subtraction.
"""

import numpy as np

from typing import Tuple, Optional, Dict, Any


# =============================================================================
# DPC Gradient Fitting for Spherical Wavefront Extraction
# =============================================================================


def _coerce_pixel_size(pixel_size: Tuple[float, float] | float) -> Tuple[float, float]:
    """Return pixel size as ``(py, px)`` in meters."""
    if isinstance(pixel_size, (float, int)):
        return float(pixel_size), float(pixel_size)
    return float(pixel_size[0]), float(pixel_size[1])

def _fit_1d_linear_weighted(
    data: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    min_valid_fraction: float = 0.5,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Perform a weighted linear fit on 1D data: ``data = slope * coords + offset``.

    Args:
        data: One-dimensional data array.
        coords: Matching coordinate array.
        weights: Weight array.
        min_valid_fraction: Minimum fraction of valid data points.

    Returns:
        ``(slope, offset)`` or ``(None, None)`` if the fit fails.
    """
    valid = np.isfinite(data)
    if np.sum(valid) < len(data) * min_valid_fraction:
        return None, None

    w = weights.copy()
    w[~valid] = 0
    if np.sum(w) < 1e-12:
        return None, None

    data_clean = data.copy()
    data_clean[~valid] = 0

    A = np.column_stack([coords, np.ones_like(coords)])
    sqrt_w = np.sqrt(w)

    try:
        coeffs, *_ = np.linalg.lstsq(
            A * sqrt_w[:, None], data_clean * sqrt_w, rcond=None
        )
        return coeffs[0], coeffs[1]
    except np.linalg.LinAlgError:
        return None, None


def _fit_dpc_rows(
    dpc_x: np.ndarray,
    x_coords: np.ndarray,
    weights_x: np.ndarray,
) -> Tuple[list, list]:
    """Fit each row of ``dpc_x`` linearly and extract slope and intercept."""
    slopes, offsets = [], []
    for i in range(dpc_x.shape[0]):
        slope, offset = _fit_1d_linear_weighted(dpc_x[i, :], x_coords, weights_x)
        if slope is not None:
            slopes.append(slope)
            offsets.append(offset)
    return slopes, offsets


def _fit_dpc_cols(
    dpc_y: np.ndarray,
    y_coords: np.ndarray,
    weights_y: np.ndarray,
) -> Tuple[list, list]:
    """Fit each column of ``dpc_y`` linearly and extract slope and intercept."""
    slopes, offsets = [], []
    for j in range(dpc_y.shape[1]):
        slope, offset = _fit_1d_linear_weighted(dpc_y[:, j], y_coords, weights_y)
        if slope is not None:
            slopes.append(slope)
            offsets.append(offset)
    return slopes, offsets


def _calculate_focus_from_curvature(
    slope: float,
    wavelength: float,
):
    """
    Compute the per-axis focus distance directly from the total DPC slope.

    For total physical DPC gradients, ``dpc = ∂phase/∂x`` or ``∂phase/∂y``,
    the fitted slope satisfies ``slope = 2a`` where ``a`` is the quadratic
    phase coefficient in ``rad/m²``. The absolute wavefront curvature is then

        ``1 / R = λ * a / π``.

    """
    a = slope / 2.0  # Quadratic coefficient [rad/m²]
    curvature = wavelength * a / np.pi  # Absolute curvature [1/m]

    if abs(curvature) > 1e-14:
        R_real = 1.0 / curvature
    else:
        R_real = np.inf

    return R_real


def fit_dpc_1d(
    dpc_x: np.ndarray,
    dpc_y: np.ndarray,
    pixel_size: Tuple[float, float],
    wavelength: float,
    use_robust: bool = True,
    weight_sigma: float = 0.4,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the low-order component of total physical DPC gradients.

    Physical background
    --------
    For a parabolic reference phase ``phase = a*x² + b*y² + tx*x + ty*y + c``,
    the physical DPC gradients are affine in each axis:

        ``g_x = ∂phase/∂x = 2a*(x - x0) = slope_x*x + offset_x``
        ``g_y = ∂phase/∂y = 2b*(y - y0) = slope_y*y + offset_y``

    Fitting those low-order DPC terms provides:
    - the per-axis curvature used for focus estimation
    - the fitted low-order DPC component used for residual subtraction

    The absolute focus distance follows
    ``1 / R_real = λ * a / π`` with ``a = slope / 2``.

    Parameters
    ----------
    dpc_x : np.ndarray
        Horizontal physical DPC gradient ``∂phase/∂x`` [rad/m].
    dpc_y : np.ndarray
        Vertical physical DPC gradient ``∂phase/∂y`` [rad/m].
    pixel_size : tuple
        Pixel size ``(py, px)`` [m]
    wavelength : float
        X-ray wavelength [m]
    use_robust : bool
        Whether to use robust fitting (median instead of mean)
    weight_sigma : float
        Gaussian weight sigma, relative to the field size
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    fit_params : dict
        Minimal fit parameters used by callers: center, slopes, quadratic
        coefficients, and fitted distances.
    dpc_x_fit : np.ndarray
        Fitted low-order DPC component of ``dpc_x`` [rad/m].
    dpc_y_fit : np.ndarray
        Fitted low-order DPC component of ``dpc_y`` [rad/m].
    dpc_x_residual : np.ndarray
        Residual horizontal DPC after subtracting the low-order fit [rad/m].
    dpc_y_residual : np.ndarray
        Residual vertical DPC after subtracting the low-order fit [rad/m].
    """
    if dpc_x.shape != dpc_y.shape:
        raise ValueError(
            f"dpc_x and dpc_y must have the same shape, got {dpc_x.shape} and {dpc_y.shape}"
        )

    H, W = dpc_x.shape
    py, px = _coerce_pixel_size(pixel_size)

    # ===== 1. Build coordinates and weights =====
    y_coords = (np.arange(H) - (H - 1) / 2.0) * py
    x_coords = (np.arange(W) - (W - 1) / 2.0) * px

    # Gaussian weights that down-weight edge pixels
    sigma_x = weight_sigma * (W * px) / 2.0
    sigma_y = weight_sigma * (H * py) / 2.0
    weights_x = np.exp(-(x_coords**2) / (2 * sigma_x**2))
    weights_y = np.exp(-(y_coords**2) / (2 * sigma_y**2))

    # ===== 2. Fit rows and columns linearly =====
    slopes_x, offsets_x = _fit_dpc_rows(dpc_x, x_coords, weights_x)
    slopes_y, offsets_y = _fit_dpc_cols(dpc_y, y_coords, weights_y)

    if len(slopes_x) == 0 or len(slopes_y) == 0:
        zeros_x = np.zeros_like(dpc_x)
        zeros_y = np.zeros_like(dpc_y)
        return {}, zeros_x, zeros_y, zeros_x.copy(), zeros_y.copy()

    # ===== 3. Aggregate fit parameters (robust median or mean) =====
    aggregate = np.median if use_robust else np.mean
    slope_x = aggregate(slopes_x)
    slope_y = aggregate(slopes_y)
    offset_x = aggregate(offsets_x)
    offset_y = aggregate(offsets_y)

    # ===== 4. Build the fitted low-order DPC components =====
    X, Y = np.meshgrid(x_coords, y_coords)

    dpc_x_fit = slope_x * X + offset_x
    dpc_y_fit = slope_y * Y + offset_y

    dpc_x_residual = dpc_x - dpc_x_fit
    dpc_y_residual = dpc_y - dpc_y_fit

    # ===== 5. Compute focus distances =====
    a_x, a_y = slope_x / 2.0, slope_y / 2.0
    R_x = _calculate_focus_from_curvature(slope_x, wavelength)
    R_y = _calculate_focus_from_curvature(slope_y, wavelength)

    # ===== 6. Recover the center position from the offsets =====
    x0 = -offset_x / slope_x if abs(slope_x) > 1e-14 else 0.0
    y0 = -offset_y / slope_y if abs(slope_y) > 1e-14 else 0.0

    # ===== 7. Build the result dictionary =====
    fit_params = {
        "x0": x0,
        "y0": y0,
        "slope_x": slope_x,
        "slope_y": slope_y,
        "a_x": a_x,
        "a_y": a_y,
        "R_x": R_x,
        "R_y": R_y,
    }

    return fit_params, dpc_x_fit, dpc_y_fit, dpc_x_residual, dpc_y_residual


def run_dpc_fitting(
    dpc_x: np.ndarray,
    dpc_y: np.ndarray,
    pixel_size: Tuple[float, float],
    wavelength: float,
    use_robust: bool = True,
    weight_sigma: float = 0.4,
) -> Dict[str, Any]:
    """
    Fit the main DPC body and return the fitted body plus residual.

    The residual is the raw DPC minus the fitted low-order DPC body.

    Returns a dictionary with:
    - ``fit_params``: low-order DPC fit metadata
    - ``dpc_fit``: fitted low-order DPC components
    - ``dpc_residual``: residual DPC components
    """
    if dpc_x.shape != dpc_y.shape:
        raise ValueError(
            f"dpc_x and dpc_y must have the same shape, got {dpc_x.shape} and {dpc_y.shape}"
        )
    if wavelength is None or not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be a finite positive value")

    pixel_size_tuple = _coerce_pixel_size(pixel_size)
    dpc_x_input = np.asarray(dpc_x)
    dpc_y_input = np.asarray(dpc_y)
    (
        fit_params,
        dpc_x_fit,
        dpc_y_fit,
        dpc_x_residual,
        dpc_y_residual,
    ) = fit_dpc_1d(
        dpc_x=dpc_x_input,
        dpc_y=dpc_y_input,
        pixel_size=pixel_size_tuple,
        wavelength=wavelength,
        use_robust=use_robust,
        weight_sigma=weight_sigma,
    )
    if not fit_params:
        raise ValueError("DPC body fitting failed; unable to identify residual")

    dpc_fit = {
        "dpc_x_fit": dpc_x_fit,
        "dpc_y_fit": dpc_y_fit,
    }
    dpc_residual = {
        "dpc_x_residual": dpc_x_residual,
        "dpc_y_residual": dpc_y_residual,
    }

    return {
        "fit_params": fit_params,
        "dpc_fit": dpc_fit,
        "dpc_residual": dpc_residual,
    }
