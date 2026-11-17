"""Detector-array corrections without file access or input-array mutation."""

import numpy as np

from .core.arrays import validate_image


def preprocess_image(image, *, dark=None, flat=None):
    """Apply dark/flat correction in float64 without changing image geometry."""
    image = validate_image(image, "detector image")
    dark = None if dark is None else validate_image(dark, "dark image")
    flat = None if flat is None else validate_image(flat, "flat image")
    for name, calibration in (("dark", dark), ("flat", flat)):
        if calibration is not None and calibration.shape != image.shape:
            raise ValueError(f"{name} image must match detector image shape")
    corrected = image.copy() if dark is None else image - dark
    if flat is not None:
        denominator = flat if dark is None else flat - dark
        if np.any(np.abs(denominator) <= 1e-8):
            raise ValueError("flat correction contains unsupported pixels")
        corrected /= denominator
    return validate_image(corrected, "corrected detector image")
