"""Minimal physical configuration; numerical controls are internal defaults."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .core.arrays import normalize_pixel_size
from .defaults import RECONSTRUCTION_DEFAULTS
from .physics.wavelength import calculate_wavelength

_PATH_KEYS = ("image_path", "dark_image_path", "flat_image_path")


def normalize_config(params):
    """Validate five physical inputs and derive optics without changing inputs."""
    if not isinstance(params, Mapping):
        raise ValueError("configuration must be a mapping")
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
    result["grating_period"] = result["g_period"] / (
        np.sqrt(2) if np.isclose(angle, 45.0) else 2
    )
    result["pattern_period"] = (
        result["grating_period"] * result["total_dist"] / result["source_dist"]
    )
    result.update({key: params.get(key) for key in _PATH_KEYS})
    result.update(RECONSTRUCTION_DEFAULTS)
    return result


def load_config(source, *, base_dir=None):
    """Load a mapping or JSON; resolve JSON image paths relative to that file.

    An explicit base_dir overrides the JSON directory. Mapping paths otherwise
    retain their working-directory meaning. Threshold/ROI keys cannot override
    the fixed online policy.
    """
    if isinstance(source, Mapping):
        raw = source
    else:
        config_path = Path(source).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if base_dir is None:
            base_dir = config_path.parent
    params = normalize_config(raw)
    for key in _PATH_KEYS:
        if params[key]:
            path = Path(params[key]).expanduser()
            if base_dir is not None and not path.is_absolute():
                path = Path(base_dir).expanduser().resolve() / path
            params[key] = str(path)
    return params
