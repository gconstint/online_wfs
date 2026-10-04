"""Shared array conventions: sampling is always ``(py, px)`` in meters."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2


def normalize_pixel_size(pixel_size) -> tuple[float, float]:
    """Validate a positive scalar or ``(py, px)`` pair without truncation."""
    values = np.asarray(pixel_size, dtype=float)
    if values.ndim == 0:
        values = np.repeat(values, 2)
    if values.shape != (2,) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("pixel_size must be a finite positive scalar or (py, px)")
    return float(values[0]), float(values[1])


def validate_image(image, name: str = "image") -> np.ndarray:
    """Return a finite, nonempty, real 2D image as floating point."""
    if np.iscomplexobj(image):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def centered_fft(image: np.ndarray) -> np.ndarray:
    """Centered orthonormal FFT used by both file and array reconstruction."""
    # Keep NumPy FFT's double precision, including integer/float32 inputs.
    dtype = np.complex128 if np.iscomplexobj(image) else np.float64
    return np.fft.fftshift(fft2(np.asarray(image, dtype=dtype), norm="ortho"))


def _validate_shape(shape):
    if len(shape) != 2 or any(
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or n <= 0
        for n in shape
    ):
        raise ValueError("shape must contain two positive integers")


def circular_domain(
    shape, pixel_size, *, center_m=(0.0, 0.0), radius_m=None, radius_fraction=1.0
):
    """Return a physical circle and its geometry on the grating grid.

    Center is (x, y), sampling is (py, px), both in meters. Coordinates follow
    the reconstruction convention (index - size/2) * sampling. Fraction scales
    the largest circle about the requested center that fits the physical FOV.
    An explicit radius takes precedence; circles are never clipped to the FOV.
    """
    _validate_shape(shape)
    py, px = normalize_pixel_size(pixel_size)
    center = np.asarray(center_m, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("circle_center_m must be finite [x, y] coordinates in meters")
    if isinstance(radius_fraction, (bool, np.bool_)):
        raise ValueError("circle_radius_fraction must lie in (0, 1]")
    fraction = float(radius_fraction)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("circle_radius_fraction must lie in (0, 1]")
    ny, nx = shape
    cx, cy = center
    maximum = min(nx * px / 2 - abs(cx), ny * py / 2 - abs(cy))
    if maximum <= 0:
        raise ValueError("circle_center_m must be inside the reconstructed FOV")
    if isinstance(radius_m, (bool, np.bool_)):
        raise ValueError("circle_radius_m must be finite and positive")
    radius = maximum * fraction if radius_m is None else float(radius_m)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("circle_radius_m must be finite and positive")
    if radius > maximum * (1 + 8 * np.finfo(float).eps):
        raise ValueError(
            "circle must fit inside the reconstructed FOV; reduce its radius"
        )
    x = (np.arange(nx) - nx / 2) * px - cx
    y = (np.arange(ny) - ny / 2) * py - cy
    mask = y[:, None] ** 2 + x[None, :] ** 2 <= radius**2
    if not np.any(mask):
        raise ValueError("circle contains no reconstructed pixels")
    return mask, {
        "shape": "circle",
        "center_m": [float(cx), float(cy)],
        "radius_m": float(radius),
        "radius_fraction": fraction if radius_m is None else None,
        "max_inscribed_radius_m": float(maximum),
        "plane": "grating",
    }
