"""Detector preprocessing without interaction or input-array mutation."""

import numpy as np
import tifffile

from .core.arrays import validate_image


def load_image(path):
    """Read one TIFF detector or calibration frame."""
    return tifffile.imread(path)


def preprocess_image(image, *, dark=None, flat=None, rotation_angle=0.0):
    """Apply dark/flat correction in float64 and a supplied fixed rotation."""
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
    corrected = validate_image(corrected, "corrected detector image")
    if abs(rotation_angle) > 1e-6:
        import cv2

        rows, columns = corrected.shape
        matrix = cv2.getRotationMatrix2D(
            ((columns - 1) / 2, (rows - 1) / 2), rotation_angle, 1.0
        )
        corrected = cv2.warpAffine(
            corrected,
            matrix,
            (columns, rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return corrected
