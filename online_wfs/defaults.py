"""Fixed online analysis policy, matching wfs_sgi 8b3d6e0 defaults.

These are implementation defaults, not detector configuration inputs.
"""

from types import MappingProxyType

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
    }
)
ZERNIKE_NUM_TERMS = 36
ZERNIKE_TUKEY_ALPHA = 0.3
