"""
Core analysis modules for the XGI wavefront sensor pipeline.
"""

from .beam_analysis import (
    calculate_and_visualize_beam,
    analyze_focus_sampling_from_beam,
)
from .dpc_preprocess import preprocess_dpc
from .grating_analysis import (
    analyze_grating_data,
    accurate_harmonic_periods,
    rotate_image_by_peaks,
    calculate_harmonic_periods,
)
from .optical_physics import (
    calculate_magnification_correction,
    calibrate_distance,
)
from .phase_analysis import dpc_integration
from .phase_fit import (
    perform_wavefront_fitting,
    plot_phase_error_profiles,
)
from .propagation import two_steps_fresnel_method
from .utils import (
    center_crop,
    image_correction,
    load_images,
    calculate_wavelength,
)

__all__ = [
    # beam_analysis
    "calculate_and_visualize_beam",
    "analyze_focus_sampling_from_beam",
    # dpc_preprocess
    "preprocess_dpc",
    # grating_analysis
    "analyze_grating_data",
    "accurate_harmonic_periods",
    "rotate_image_by_peaks",
    "calculate_harmonic_periods",
    # optical_physics
    "calculate_magnification_correction",
    "calibrate_distance",
    # phase_analysis
    "dpc_integration",
    # phase_fit
    "perform_wavefront_fitting",
    "plot_phase_error_profiles",
    # propagation
    "two_steps_fresnel_method",
    # utils
    "center_crop",
    "image_correction",
    "load_images",
    "calculate_wavelength",
]