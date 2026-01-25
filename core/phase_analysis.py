from os import cpu_count

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

# Module-level constant for parallel FFT
_CPU_COUNT = cpu_count() or 4


def _reflect_and_pad_gradient_fields(delta_x, delta_y):
    """
    Reflect and pad gradient fields (preserve physical sign correctness).
    Optimized version: uses pre-allocated arrays and direct assignment.
    """
    h, w = delta_x.shape
    h2, w2 = h * 2, w * 2

    # Pre-allocate output arrays
    gx = np.empty((h2, w2), dtype=delta_x.dtype)
    gy = np.empty((h2, w2), dtype=delta_y.dtype)

    # X-direction gradient filling
    # Upper-left (original)
    gx[:h, :w] = delta_x
    # Lower-left (y-flipped)
    gx[h:, :w] = delta_x[::-1, :]
    # Upper-right (x-flipped, negated)
    gx[:h, w:] = -delta_x[:, ::-1]
    # Lower-right (xy-flipped, negated)
    gx[h:, w:] = -delta_x[::-1, ::-1]

    # Y-direction gradient filling
    # Upper-left (original)
    gy[:h, :w] = delta_y
    # Lower-left (y-flipped, negated)
    gy[h:, :w] = -delta_y[::-1, :]
    # Upper-right (x-flipped)
    gy[:h, w:] = delta_y[:, ::-1]
    # Lower-right (xy-flipped, negated)
    gy[h:, w:] = -delta_y[::-1, ::-1]

    return gx, gy


def _crop_to_central_quarter(array):
    """Crop back to upper-left original region (using slicing, more efficient)."""
    h, w = array.shape
    return array[: h // 2, : w // 2]


def _ensure_concave_shape(phase):
    """
    Force phase to concave-up (bowl) shape.
    Determines curvature direction by parabolic fitting of center cross-sections.
    """
    H, W = phase.shape

    # Extract center row and center column
    mid_row = phase[H // 2, :]
    mid_col = phase[:, W // 2]

    # Use polyfit for fast quadratic fitting (y = ax^2 + bx + c)
    # Only the quadratic coefficient 'a' matters
    x = np.arange(W)
    y = np.arange(H)

    coeff_x = np.polyfit(x, mid_row, 2)
    coeff_y = np.polyfit(y, mid_col, 2)

    # Combined check: if sum of curvatures in both directions < 0, flip required
    if (coeff_x[0] + coeff_y[0]) < 0:
        return -phase

    return phase


def fc_method(delta_x, delta_y, reflected_pad=True):
    """
    Frankot-Chellappa integration algorithm (optimized version).
    Uses scipy.fft and broadcasting for improved performance.
    """
    # 1. Reflected padding
    if reflected_pad:
        gx, gy = _reflect_and_pad_gradient_fields(delta_x, delta_y)
    else:
        gx, gy = delta_x, delta_y

    rows, cols = gx.shape

    # 2. Frequency grid (using scipy.fft.fftfreq and broadcasting)
    wx = fftfreq(cols) * (2 * np.pi)  # 1D array (cols,)
    wy = fftfreq(rows) * (2 * np.pi)  # 1D array (rows,)

    # 3. FFT with multi-threading
    fx = fft2(gx, workers=_CPU_COUNT)
    fy = fft2(gy, workers=_CPU_COUNT)

    # 4. Frequency-domain integration (using broadcasting instead of meshgrid)
    # wx[None, :] -> (1, cols), wy[:, None] -> (rows, 1)
    numerator = -1j * (wx[None, :] * fx + wy[:, None] * fy)
    denominator = wx[None, :] ** 2 + wy[:, None] ** 2

    # Handle singularity (DC component)
    denominator[0, 0] = 1.0
    numerator[0, 0] = 0.0

    # 5. IFFT with multi-threading and take real part
    phase_2D = np.real(ifft2(numerator / denominator, workers=_CPU_COUNT))

    # 6. Crop
    if reflected_pad:
        phase_2D = _crop_to_central_quarter(phase_2D)

    return phase_2D


def dpc_integration(dpc_x, dpc_y, ensure_concave=True):
    """
    DPC integration main function.

    Args:
        dpc_x, dpc_y: Gradient maps.
        ensure_concave (bool): Whether to force output to concave-up (bowl) shape.
                               For converging beams, phase is typically concave.

    Returns:
        ndarray: Integrated reconstructed phase map.
    """
    # 1. Perform integration
    phase = fc_method(dpc_x, dpc_y, reflected_pad=True)

    # 2. Remove DC component
    phase -= np.mean(phase)

    # 3. Enforce shape constraint
    if ensure_concave:
        phase = _ensure_concave_shape(phase)

    return phase
