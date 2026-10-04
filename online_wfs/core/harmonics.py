"""Absolute carrier phases using shared physical FFT crop geometry."""

import numpy as np
from numpy.fft import ifft2, ifftshift

from .dpc_conversion import detector_to_wavefront_pixel_size
from .harmonic_peaks import (
    HARMONIC_NAMES,
    calculate_harmonic_periods,
    get_harmonic_geometry,
)


def _remove_carrier_ramp(image, geometry, dc):
    """Compensate crop displacement while retaining the theoretical carrier."""
    row_offset, column_offset = geometry["carrier_offset"]
    carrier = np.asarray(geometry["theoretical_peak"]) - dc
    anchor = carrier - np.rint(carrier)
    rows, columns = image.shape
    row = np.arange(rows, dtype=float) / rows
    column = np.arange(columns, dtype=float) / columns
    ramp = (
        2
        * np.pi
        * (
            row_offset * row[:, None]
            + column_offset * column[None, :]
            - 0.5 * np.sum(anchor)
        )
    )
    return image * np.exp(-1j * ramp)


def extract_harmonics(detector_fft, params):
    """Extract DC and positive horizontal/vertical harmonics in absolute mode."""
    periods = calculate_harmonic_periods(
        detector_fft.shape, params["pixel_size"], params["pattern_period"]
    )
    images, geometries = [], {}
    dc = np.asarray(detector_fft.shape) // 2
    for name in HARMONIC_NAMES:
        geometry = get_harmonic_geometry(
            detector_fft,
            periods,
            name,
            peak_mode=params["harmonic_peak_mode"],
            search_region=None,
        )
        y0, y1, x0, x1 = geometry["window_bounds"]
        image = ifft2(ifftshift(detector_fft[y0:y1, x0:x1]), norm="ortho")
        images.append(_remove_carrier_ramp(image, geometry, dc))
        geometries[name] = geometry
    int00, int01, int10 = map(np.abs, images)
    detector_sampling = tuple(
        params["pixel_size"][axis] * detector_fft.shape[axis] / int00.shape[axis]
        for axis in (0, 1)
    )
    magnification = params["total_dist"] / params["source_dist"]
    # Intensity/carrier phase comes from the detector; wavefront sampling maps to the grating.
    return {
        "int00": int00,
        "int01": int01,
        "int10": int10,
        "support_amplitude_x": int01,
        "support_amplitude_y": int10,
        "wrapped_phase_x_rad": np.angle(images[1]),
        "wrapped_phase_y_rad": np.angle(images[2]),
        "detector_pixel_size": detector_sampling,
        "wavefront_pixel_size": detector_to_wavefront_pixel_size(
            detector_sampling, magnification
        ),
        "harmonic_peak_diagnostics": {
            "peak_mode": params["harmonic_peak_mode"],
            "detector": {"extraction_geometry": geometries},
        },
    }
