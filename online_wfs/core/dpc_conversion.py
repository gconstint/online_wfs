"""Physical conversions between carrier phase and wavefront gradients."""

from typing import Sequence

import numpy as np


def carrier_phase_to_wavefront_gradient(
    carrier_phase: np.ndarray,
    detector_pattern_period: float,
    propagation_distance: float,
    wavelength: float,
    harmonic_peak_sign: int = 1,
) -> np.ndarray:
    """Convert a demodulated carrier phase to ``d(phi)/dx`` or ``d(phi)/dy``.

    For a positive-frequency carrier, a local refraction angle translates the
    detector pattern and produces

    ``carrier_phase = -wavelength * distance / period * wavefront_gradient``.

    ``harmonic_peak_sign`` is ``+1`` for the positive Fourier peak and ``-1``
    for the conjugate negative peak.
    """
    if harmonic_peak_sign not in (-1, 1):
        raise ValueError("harmonic_peak_sign must be +1 or -1")
    if detector_pattern_period <= 0:
        raise ValueError("detector_pattern_period must be positive")
    if propagation_distance <= 0:
        raise ValueError("propagation_distance must be positive")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")

    scale = detector_pattern_period / (wavelength * propagation_distance)
    return -harmonic_peak_sign * np.asarray(carrier_phase, dtype=float) * scale


def detector_to_wavefront_pixel_size(
    detector_pixel_size: Sequence[float], magnification: float
) -> tuple[float, float]:
    """Map detector-plane sampling ``(py, px)`` back to the grating plane."""
    if magnification <= 0:
        raise ValueError("magnification must be positive")
    return (
        float(detector_pixel_size[0]) / magnification,
        float(detector_pixel_size[1]) / magnification,
    )
