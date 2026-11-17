"""Wavepy-style harmonic crops with a circular super-Gaussian filter."""

import numpy as np
from scipy.fft import ifft2, rfft2

from .. import defaults
from .dpc_conversion import detector_to_wavefront_pixel_size
from .harmonic_peaks import (
    calculate_harmonic_periods,
    calculate_peak_index,
    harmonic_window_bounds,
)


def demodulation_radius(shape, periods, rotation_deg, requested=None):
    """Keep each low-pass sideband separate from DC and other first harmonics."""
    periods = np.asarray(periods, float)
    # Validate period/rotation using the same carrier geometry as extraction.
    if periods.shape != (2,) or not np.all(np.isfinite(periods) & (periods > 2)):
        raise ValueError(
            "carrier periods must contain two finite values above 2 pixels"
        )
    angle = np.deg2rad(rotation_deg)
    if not np.isfinite(angle):
        raise ValueError("carrier rotation must be finite")
    vectors = (
        np.array([[np.sin(angle), np.cos(angle)], [np.cos(angle), -np.sin(angle)]])
        / periods[:, None]
    )
    if np.any(np.abs(vectors) >= 0.5):
        raise ValueError("carrier exceeds the detector Nyquist frequency")
    lobes = np.vstack((np.zeros(2), vectors, -vectors))
    distances = []
    for carrier in vectors:
        delta = lobes - carrier
        delta -= np.rint(delta)
        spacing = np.linalg.norm(delta * np.asarray(shape), axis=1)
        distances.extend(spacing[spacing > 1e-8])
    nearest = min(distances)
    radius = 0.45 * nearest if requested is None else float(requested)
    if not np.isfinite(radius) or radius <= 0 or radius > 0.8 * nearest:
        raise ValueError(
            "demodulation_radius_px overlaps another carrier; use null/None for automatic bandwidth"
        )
    return radius


def extract_harmonics_from_image(image, params, *, include_dc=True):
    """FFT the detector once, then filter, pad and IFFT the harmonic crops."""
    if np.iscomplexobj(image):
        raise ValueError("harmonic extraction requires a real detector image")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 3 or not np.all(np.isfinite(image)):
        raise ValueError("harmonic extraction requires a finite 2D detector image")
    # Retain the existing float32 input rounding and double-precision transform.
    # Only small selected windows need reconstruction of the Hermitian half.
    spectrum = rfft2(image.astype(float), norm="ortho", workers=defaults.cpu_workers())
    detector_shape = image.shape
    pixels = np.asarray(params["pixel_size"], float)
    periods = params["pattern_period"] / pixels[::-1]
    radius = demodulation_radius(detector_shape, periods, 0.0)
    sign = params["harmonic_peak_sign"]
    if isinstance(sign, (bool, np.bool_)) or sign not in (-1, 1):
        raise ValueError("harmonic_peak_sign must be +1 or -1")
    factor = defaults.HARMONIC_OVERSAMPLING_FACTOR
    if (
        isinstance(factor, (bool, np.bool_))
        or not isinstance(factor, (int, np.integer))
        or factor < 1
    ):
        raise ValueError("HARMONIC_OVERSAMPLING_FACTOR must be a positive integer")
    metadata, shape, normalization, windows = _plan_small_grid(
        detector_shape,
        params,
        pixels,
        radius,
        sign,
        include_dc,
        int(factor),
    )
    harmonics = dict(metadata)
    detector_shape = np.asarray(detector_shape)
    for name, axis, bounds, filter_shift, correction in windows:
        y0, y1, x0, x1 = bounds
        ky = np.arange(y0, y1) - detector_shape[0] // 2
        kx = np.arange(x0, x1) - detector_shape[1] // 2
        conjugate = kx < 0
        rows = np.where(conjugate[None, :], -ky[:, None], ky[:, None])
        cropped = spectrum[rows % detector_shape[0], np.abs(kx)[None, :]].copy()
        cropped[:, conjugate] = cropped[:, conjugate].conj()
        cropped *= filter_shift
        if factor != 1:
            padded = np.zeros(shape, dtype=cropped.dtype)
            start = np.asarray(shape) // 2 - np.asarray(cropped.shape) // 2
            padded[
                start[0] : start[0] + cropped.shape[0],
                start[1] : start[1] + cropped.shape[1],
            ] = cropped
            cropped = padded
        field = ifft2(
            np.fft.ifftshift(cropped), norm="ortho", workers=defaults.cpu_workers()
        )
        field *= normalization
        field *= correction
        harmonics[f"int{name}"] = np.abs(field)
        if axis is not None:
            harmonics[f"wrapped_phase_{axis}_rad"] = np.angle(field)
    return harmonics


def _plan_small_grid(detector_shape, params, pixels, radius, sign, include_dc, factor):
    """Build crops, filters and physical phase corrections for the current image."""
    detector_shape = np.asarray(detector_shape)
    spacing = np.asarray(
        calculate_harmonic_periods(detector_shape, pixels, params["pattern_period"])
    )
    # wavepy's example truncates spacings before using them as crop centers.
    # Keep the unrounded spacings for filtering and physical phase correction.
    crop_spacing = spacing.astype(int)
    crop_shape = 2 * (crop_spacing // 2)
    if np.any(crop_shape < 4):
        raise ValueError(
            "harmonic grid is smaller than 4x4; use a larger detector crop"
        )
    shape = crop_shape * int(factor)
    step = detector_shape / shape
    shift = (step - 1) / 2
    detector_pixels = pixels * step
    magnification = params["total_dist"] / params["source_dist"]
    harmonics = {
        "wavefront_pixel_size": detector_to_wavefront_pixel_size(
            detector_pixels, magnification
        ),
        # Preserve the physical 16-detector-pixel fit margin on the output grid.
        "fit_edge_margin_yx": tuple(np.ceil(16 / step).astype(int)),
    }
    ly = np.arange(crop_shape[0]) - crop_shape[0] // 2
    lx = np.arange(crop_shape[1]) - crop_shape[1] // 2
    # A direct padded IFFT samples n=j*N/m. Shift it to the centered grid
    # n=j*N/m+(N/m-1)/2, matching the coordinate convention used downstream.
    center_shift = np.exp(
        2j
        * np.pi
        * (
            ly[:, None] * shift[0] / detector_shape[0]
            + lx[None, :] * shift[1] / detector_shape[1]
        )
    )
    ny = np.arange(shape[0]) * step[0] + shift[0]
    nx = np.arange(shape[1]) * step[1] + shift[1]
    normalization = np.sqrt(np.prod(shape) / np.prod(detector_shape))
    windows = []
    for name, axis, order in (
        ("00", None, (0, 0)),
        ("01", "x", (0, sign)),
        ("10", "y", (sign, 0)),
    ):
        if axis is None and not include_dc:
            continue
        peak = np.asarray(calculate_peak_index(*order, *detector_shape, *spacing))
        crop_peak = calculate_peak_index(*order, *detector_shape, *crop_spacing)
        bounds = harmonic_window_bounds(detector_shape, crop_peak, crop_spacing)
        y0, y1, x0, x1 = bounds
        center = np.array([(y0 + y1) // 2, (x0 + x1) // 2])
        fy = np.arange(y0, y1) - peak[0]
        fx = np.arange(x0, x1) - peak[1]
        # Preserve the circular eighth-order super-Gaussian, centered on the
        # exact carrier rather than its truncated FFT crop center.
        weights = np.exp(
            -0.5 * ((fy[:, None] ** 2 + fx[None, :] ** 2) / radius**2) ** 4
        )
        carrier = peak - detector_shape // 2
        delta = peak - center
        phase = (
            2
            * np.pi
            * (
                -delta[0] * ny[:, None] / detector_shape[0]
                - delta[1] * nx[None, :] / detector_shape[1]
                + np.sum(carrier * (detector_shape - 1) / (2 * detector_shape))
            )
        )
        windows.append((name, axis, bounds, weights * center_shift, np.exp(1j * phase)))
    return harmonics, tuple(shape), normalization, windows
