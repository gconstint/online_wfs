"""Reconstruction sampling policy and generic analysis choices.

These are implementation defaults, not detector configuration inputs.
"""

from numbers import Integral
from os import cpu_count
from types import MappingProxyType

# CPU budget for FFT/DCT transforms and independent carrier unwrapping.
CPU_WORKERS = 4


def cpu_workers():
    """Return the positive worker budget, capped at the available CPU count."""
    if (
        isinstance(CPU_WORKERS, bool)
        or not isinstance(CPU_WORKERS, Integral)
        or CPU_WORKERS < 1
    ):
        raise ValueError("CPU_WORKERS must be a positive integer")
    return min(int(CPU_WORKERS), cpu_count() or 1)


# Per-axis frequency-domain padding after harmonic cropping; 1 disables padding.
HARMONIC_OVERSAMPLING_FACTOR = 1

RECONSTRUCTION_DEFAULTS = MappingProxyType(
    {
        "domain": "circle",
        "support_threshold": 0.01,
        "circle_center_m": (0.0, 0.0),
        "circle_radius_m": None,
        "circle_radius_fraction": 1.0,
        "dpc_fit_valid_fraction": 0.8,
        "dpc_fit_use_robust": True,
        "harmonic_peak_mode": "theoretical",
        "harmonic_peak_sign": 1,
        "integration_method": "fc_cg",
        "phase_cleanup": "tilt_xy",
    }
)
ZERNIKE_NUM_TERMS = 36
ZERNIKE_TUKEY_ALPHA = 0.3
