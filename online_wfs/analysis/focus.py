"""Estimate grating-to-focus distances with restored curvature and Gaussian q."""

import numpy as np


def focus_distance_from_dpc(
    slope_x: float,
    slope_y: float,
    detector_fwhm_x: float,
    detector_fwhm_y: float,
    wavelength: float,
    source_distance: float,
    propagation_distance: float,
) -> tuple[float, float]:
    """Estimate signed grating-to-focus distances using the Gaussian-q model.

    The inferred focus is the Gaussian waist. For a non-Gaussian beam, this is
    an equivalent Gaussian estimate, not a minimum found by field propagation.

    Demodulation removes the nominal spherical carrier, so a fitted slope
    measures a curvature deviation, not the full wavefront curvature. With
    M = 1 + L/R0 and delta_kappa = wavelength * slope / (2*pi), restore

        kappa_detector = 1/(R0 + L) + delta_kappa/M**2.

    The M**2 factor accounts for the carrier-phase scale and nominal grating
    coordinates. Combine this full curvature with detector-plane intensity
    FWHM, then back-propagate q by L: q_grating = q_detector - L. No fitted
    residual radius is inverted, so zero residual slopes remain well-defined.
    Returned X/Y distances are Re(q_grating) in meters: positive for a waist
    upstream of the grating, negative for a waist downstream. Pipeline exports
    their absolute values as the grating-to-focus distance magnitudes.
    """
    geometry = np.asarray([source_distance, propagation_distance], dtype=float)
    slopes = np.asarray([slope_x, slope_y], dtype=float)
    if not np.all(np.isfinite(geometry) & (geometry > 0)):
        raise ValueError("source and propagation distances must be finite and positive")
    if not np.all(np.isfinite(slopes)):
        raise ValueError("DPC slopes must be finite")
    magnification = 1 + propagation_distance / source_distance
    curvature = 1 / (source_distance + propagation_distance) + wavelength * slopes / (
        2 * np.pi * magnification**2
    )
    radii = np.divide(1.0, curvature, out=np.full(2, np.inf), where=curvature != 0)
    distances = gaussian_waist_from_curvature(
        *radii, detector_fwhm_x, detector_fwhm_y, wavelength
    )
    # Back-propagate q from the detector to the grating by propagation_distance.
    # Re(q) is positive for an upstream waist and negative for a downstream waist.
    return tuple(distance - propagation_distance for distance in distances)


def gaussian_waist_from_curvature(
    curvature_radius_x: float,
    curvature_radius_y: float,
    beam_fwhm_x: float,
    beam_fwhm_y: float,
    wavelength: float,
) -> tuple[float, float]:
    """Infer Gaussian waist positions relative to the supplied measurement plane.

    The supplied FWHM values and curvature radii must refer to the same plane.
    Per axis, ``1/q = 1/R - i*lambda/(pi*w^2)`` with
    ``w = FWHM/sqrt(2*ln(2))``. The real part of ``q`` is the signed distance
    from the Gaussian waist to that plane; it is generally not equal to ``R``.
    """
    values = (
        curvature_radius_x,
        curvature_radius_y,
        beam_fwhm_x,
        beam_fwhm_y,
        wavelength,
    )
    if any(np.isnan(value) for value in values):
        raise ValueError("curvature, beam FWHM, and wavelength must not be NaN")
    if not all(np.isfinite(value) for value in (*values[2:4], wavelength)):
        raise ValueError("beam FWHM and wavelength must be finite")
    if beam_fwhm_x <= 0 or beam_fwhm_y <= 0 or wavelength <= 0:
        raise ValueError("beam FWHM and wavelength must be positive")

    def per_axis(radius: float, fwhm: float) -> float:
        if abs(radius) <= 1e-14:
            raise ValueError("curvature radius must be non-zero")
        beam_radius = fwhm / np.sqrt(2.0 * np.log(2.0))
        inverse_radius = 0.0 if np.isinf(radius) else 1.0 / radius
        inverse_q = inverse_radius - 1j * wavelength / (np.pi * beam_radius**2)
        q_parameter = 1.0 / inverse_q
        return float(np.real(q_parameter))

    x = per_axis(float(curvature_radius_x), float(beam_fwhm_x))
    y = per_axis(float(curvature_radius_y), float(beam_fwhm_y))
    return x, y
