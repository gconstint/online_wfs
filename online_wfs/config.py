"""Configuration for the array-only wavefront API.

Required: p_energy [eV], pixel_size [m, scalar or (py, px)],
grating_period [m, nominal], source_dist [m], det2sample [m].
Optional: grating_angle=45.0 [degrees], integration_method="fc_cg",
phase_cleanup="tilt_xy".
Derived: wavelength, total_dist, g_period (effective), pattern_period.
"""

from collections.abc import Mapping

import numpy as np

from .core.arrays import normalize_pixel_size
from .defaults import RECONSTRUCTION_DEFAULTS
from .physics.wavelength import calculate_wavelength


def normalize_config(params: Mapping) -> dict:
    """Validate inputs and return a new mapping with defaults and derived optics.

    Already normalized mappings are accepted. Derived values and fixed internal
    defaults are recomputed; the input mapping is never modified.
    """
    if not isinstance(params, Mapping):
        raise ValueError("configuration must be a mapping")

    # Physical inputs: energy in eV, lengths in meters.
    result = {}
    for key in ("p_energy", "grating_period", "source_dist", "det2sample"):
        raw = params[key]
        value = float(raw)
        if isinstance(raw, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = value
    result["pixel_size"] = normalize_pixel_size(params["pixel_size"])
    angle = float(params.get("grating_angle", 45.0))
    if not np.isfinite(angle):
        raise ValueError("grating_angle must be finite degrees")
    result["grating_angle"] = angle

    # Public reconstruction choices; other numerical settings stay internal.
    result.update(RECONSTRUCTION_DEFAULTS)
    for key, choices in (
        ("phase_cleanup", ("none", "tilt_xy")),
        ("integration_method", ("fc_cg", "fc")),
    ):
        value = params.get(key, result[key])
        if value not in choices:
            raise ValueError(f"{key} must be one of {choices}")
        result[key] = value

    # Derived optics are always calculated from the physical inputs above.
    result["wavelength"] = calculate_wavelength(result["p_energy"])
    result["total_dist"] = result["source_dist"] + result["det2sample"]
    result["g_period"] = result["grating_period"] * (
        np.sqrt(0.5) if np.isclose(angle, 45.0) else 0.5
    )
    result["pattern_period"] = result["g_period"] * (
        result["total_dist"] / result["source_dist"]
    )
    unsupported = params.keys() - result.keys()
    if unsupported:
        names = ", ".join(sorted(map(str, unsupported)))
        raise ValueError(f"unsupported configuration parameters: {names}")
    return result
