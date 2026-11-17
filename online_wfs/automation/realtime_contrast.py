"""Measure fringe contrast at the current grating-to-detector distance."""

from collections.abc import Callable

import numpy as np

from ..physics.contrast import measure_contrast


def measure_realtime_contrast(
    image: np.ndarray,
    grating_detector_dist_m: float | Callable[[], float],
    *,
    pixel_size: float | tuple[float, float],
    grating_period: float,
) -> float:
    """Return the mean horizontal and vertical fringe visibility for one frame.

    Pass a 2D detector frame for an aligned 45-degree checkerboard grating.
    The distance can be a reading in meters for this frame or a zero-argument
    reader called once per frame. Use a reading synchronized with the exposure
    if the stage is moving. The source-to-grating distance is fixed at 800 m
    to approximate plane-wave illumination.

    Pass the detector pixel size in meters as a scalar or (vertical, horizontal)
    pair and the nominal grating period in meters. The return value is a
    dimensionless ratio (0.12 corresponds to 12%).
    """
    if callable(grating_detector_dist_m):
        distance_m = grating_detector_dist_m()
    else:
        distance_m = grating_detector_dist_m
    return measure_contrast(
        image,
        pixel_size=pixel_size,
        grating_period=grating_period,
        source_grating_dist=800.0,
        grating_detector_dist=distance_m,
    ).contrast
