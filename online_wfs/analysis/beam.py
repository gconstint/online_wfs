"""Gaussian fitting of horizontal and vertical DC-harmonic profiles."""

import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, amplitude, mean, sigma, offset):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2)) + offset


def _fit_profile(profile, coordinates):
    """Use the SGI initialization/bounds, with an FWHM fallback on fit failure."""
    edge_count = max(1, int(len(profile) * 0.1))
    background = np.median(np.r_[profile[:edge_count], profile[-edge_count:]])
    normalized = profile - background
    peak = np.max(normalized)
    if peak > 0:
        normalized /= peak
    peak_index = np.argmax(normalized)
    span = abs(coordinates[-1] - coordinates[0])
    sigma = (
        span
        if peak_index < len(profile) * 0.2 or peak_index > len(profile) * 0.8
        else span / 8
    )
    try:
        fitted, _ = curve_fit(
            _gaussian,
            coordinates,
            normalized,
            p0=[1.0, coordinates[peak_index], sigma, 0.0],
            bounds=(
                [0.5, coordinates[0] - span * 0.5, span * 0.001, -0.2],
                [1.5, coordinates[-1] + span * 0.5, span * 0.8, 0.2],
            ),
            maxfev=10000,
        )
        return float(fitted[1]), float(2 * np.sqrt(2 * np.log(2)) * abs(fitted[2]))
    except (RuntimeError, ValueError):
        indices = np.flatnonzero(normalized >= np.max(normalized) * 0.5)
        fwhm = (
            abs(coordinates[indices[-1]] - coordinates[indices[0]])
            if len(indices) > 1
            else span * 0.1
        )
        return 0.0, float(fwhm)


def calc_beam_size(intensity, pixel_size):
    """Return detector-plane position (x,y) and Gaussian FWHM in meters."""
    rows, columns = intensity.shape
    py, px = pixel_size
    # Fit in micrometers to retain upstream numerical conditioning.
    x = (np.arange(columns) - columns / 2) * px * 1e6
    y = (np.arange(rows) - rows / 2) * py * 1e6
    x0, fwhm_x = _fit_profile(np.mean(intensity, axis=0), x)
    y0, fwhm_y = _fit_profile(np.mean(intensity, axis=1), y)
    return (x0 * 1e-6, y0 * 1e-6), {"fwhm_x": fwhm_x * 1e-6, "fwhm_y": fwhm_y * 1e-6}
