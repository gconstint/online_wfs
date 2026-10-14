"""Minimal physical configuration; numerical controls are internal defaults."""

from collections.abc import Mapping

import numpy as np

from .core.arrays import normalize_pixel_size
from .defaults import RECONSTRUCTION_DEFAULTS
from .physics.wavelength import calculate_wavelength


def normalize_config(params):
    """Validate five physical inputs and derive optics without changing inputs."""
    if not isinstance(params, Mapping):
        raise ValueError("configuration must be a mapping")
    if any(
        params.get(key)
        for key in (
            "image_path",
            "dark_image_path",
            "flat_image_path",
            "flat_valid_mask_path",
        )
    ):
        raise ValueError(
            "file paths are unsupported; supply physical parameters and image arrays"
        )
    if any(
        key in params
        for key in (
            "reference",
            "reference_image",
            "reference_fft",
            "reference_image_path",
            "ref_image_path",
        )
    ):
        raise ValueError(
            "reference images are unsupported; this package uses absolute mode"
        )
    if params.get("mode", "absolute") != "absolute":
        raise ValueError("only absolute mode is supported")
    result = {}
    for key in ("p_energy", "g_period", "source_dist", "det2sample"):
        if key not in params:
            raise KeyError(f"missing configuration parameter: {key}")
        value = params[key]
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
        result[key] = float(value)
    if "pixel_size" not in params:
        raise KeyError("missing configuration parameter: pixel_size")
    result["pixel_size"] = normalize_pixel_size(params["pixel_size"])
    angle = float(params.get("g_angle", 45.0))
    rotation = float(params.get("rotation_angle", 0.0))
    if not np.isfinite(angle) or not np.isfinite(rotation):
        raise ValueError("g_angle and rotation_angle must be finite degrees")
    result.update(g_angle=angle, rotation_angle=rotation)
    result["wavelength"] = calculate_wavelength(result["p_energy"])
    result["total_dist"] = result["source_dist"] + result["det2sample"]
    result["grating_period"] = result["g_period"] * (
        np.sqrt(0.5) if np.isclose(angle, 45.0) else 0.5
    )
    result["pattern_period"] = result["grating_period"] * (
        result["total_dist"] / result["source_dist"]
    )
    result.update(RECONSTRUCTION_DEFAULTS)
    offsets = np.asarray(params.get("carrier_phase_offset_rad", (0.0, 0.0)), float)
    if offsets.shape != (2,) or not np.all(np.isfinite(offsets)):
        raise ValueError(
            "carrier_phase_offset_rad must contain two finite [x,y] radians"
        )
    # Numerical phase origins may describe a crop of a larger detector frame.
    result["carrier_phase_offset_rad"] = tuple(map(float, offsets))
    for key, choices in (
        ("phase_cleanup", ("none", "tilt_xy")),
        ("beam_width_method", ("gaussian_profile", "gaussian_envelope")),
    ):
        value = params.get(key, result[key])
        if value not in choices:
            raise ValueError(f"{key} must be one of {choices}")
        result[key] = value
    return result
