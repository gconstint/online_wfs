"""Gaussian detector-envelope fitting with an additive intensity background."""

import numpy as np
from scipy.optimize import curve_fit


def fit_detector_fwhm(image, pixel_size_yx, valid_mask=None):
    """Fit Gaussian marginal envelopes with an additive constant background.

    Use corrected intensity and detector sampling, not a wavefront support mask.
    Invalid detector pixels are excluded from each marginal mean. No widths are
    invented if the beam envelope cannot be resolved in the recorded ROI.
    """
    image = np.asarray(image, float)
    pixels = np.asarray(pixel_size_yx, float)
    if image.ndim != 2 or min(image.shape) < 5:
        raise ValueError("beam envelope requires a detector image of at least 5x5")
    if pixels.shape != (2,) or not np.all(np.isfinite(pixels) & (pixels > 0)):
        raise ValueError("detector pixels must be positive finite [y,x]")
    valid = np.isfinite(image)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, bool)
        if mask.shape != image.shape:
            raise ValueError("detector intensity and validity mask must match")
        valid &= mask
    widths = []
    for axis, dimension, pixel in (("x", 0, pixels[1]), ("y", 1, pixels[0])):
        counts = valid.sum(axis=dimension)
        profile = np.divide(
            np.where(valid, image, 0).sum(axis=dimension),
            counts,
            out=np.full(counts.shape, np.nan),
            where=counts > 0,
        )
        finite = np.isfinite(profile)
        if finite.sum() < 5:
            raise ValueError(f"too few intensity samples along {axis}")
        x = np.arange(profile.size, dtype=float)[finite]
        values = profile[finite]
        baseline = float(values.min())
        contrast = float(np.ptp(values))
        if contrast <= max(float(np.max(abs(values))), 1.0) * 1e-8:
            raise ValueError(f"no resolved beam envelope along {axis}")
        values = (values - baseline) / contrast
        weight = np.maximum(values, 0)
        center = float(np.sum(x * weight) / weight.sum())
        sigma = float(np.sqrt(np.sum(weight * (x - center) ** 2) / weight.sum()))
        span = float(x[-1] - x[0])

        def gaussian(coordinate, amplitude, mean, width, offset):
            return (
                amplitude * np.exp(-0.5 * ((coordinate - mean) / width) ** 2) + offset
            )

        fit, _ = curve_fit(
            gaussian,
            x,
            values,
            p0=(1.0, center, np.clip(sigma, 0.5, span), 0.0),
            bounds=([0, x[0] - span, 0.1, -1], [3, x[-1] + span, 2 * span, 1]),
            maxfev=10000,
        )
        if fit[2] >= 1.99 * span or fit[0] <= 1e-8:
            raise ValueError(f"beam width is unconstrained along {axis}")
        widths.append(float(2 * np.sqrt(2 * np.log(2)) * fit[2] * pixel))
    return tuple(widths)
