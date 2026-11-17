"""Geometric grating-to-focus distances from restored DPC curvature."""

import numpy as np


def focus_distance_from_dpc(
    slope_x: float,
    slope_y: float,
    wavelength: float,
    source_distance: float,
    propagation_distance: float,
) -> tuple[float, float]:
    """Return signed geometric curvature distances, without beam-width fitting.

    Demodulation removes the nominal spherical carrier, so a fitted slope
    measures a curvature deviation, not the full wavefront curvature. With
    M = 1 + L/R0 and delta_kappa = wavelength * slope / (2*pi), restore

        kappa_detector = 1/(R0 + L) + delta_kappa/M**2.

    The M**2 factor converts the fitted nominal grating-coordinate slopes to
    detector curvature. Return 1/kappa_detector - L, positive upstream of the
    grating and negative downstream. Zero curvature returns infinity. These
    are geometric curvature distances; Pipeline exports their magnitudes.
    """
    geometry = np.asarray([source_distance, propagation_distance], dtype=float)
    slopes = np.asarray([slope_x, slope_y], dtype=float)
    if not np.all(np.isfinite(geometry) & (geometry > 0)):
        raise ValueError("source and propagation distances must be finite and positive")
    if not np.all(np.isfinite(slopes)):
        raise ValueError("DPC slopes must be finite")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive")
    magnification = 1 + propagation_distance / source_distance
    curvature = 1 / (source_distance + propagation_distance) + wavelength * slopes / (
        2 * np.pi * magnification**2
    )
    radii = np.divide(1.0, curvature, out=np.full(2, np.inf), where=curvature != 0)
    return tuple(map(float, radii - propagation_distance))
