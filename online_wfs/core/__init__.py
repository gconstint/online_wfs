"""
Core analysis modules for the XGI wavefront sensor pipeline.
"""

from .beam_analysis import (
    calc_beam_size,
    calc_focus_by_back_prop,
)
from .grating_analysis import (
    analyze_grating_data,
    accurate_harmonic_periods,
    calculate_harmonic_periods,
)
from .phase_analysis import dpc_integration
from .phase_fit import (
    perform_wavefront_fitting,
    plot_phase_error_profiles,
    preprocess_phase_for_fitting,
    find_wavefront_center,
    fit_parabolic_phase,
    fit_parabolic_phase_fast,
    plot_phase_fit_results,
)
from .propagation import two_steps_fresnel_method
from .utils import (
    center_crop,
    image_correction,
    calculate_wavelength,
    load_images,
    calculate_magnification_correction,
)
from .zernike_analysis import (
    analyze_and_visualize_zernike,
    perform_zernike_analysis,
)


__all__ = [
    # beam_analysis
    "calc_beam_size",
    "calc_focus_by_back_prop",
    # grating_analysis
    "analyze_grating_data",
    "accurate_harmonic_periods",
    "calculate_harmonic_periods",
    # utils
    "calculate_magnification_correction",
    # phase_analysis
    "dpc_integration",
    # phase_fit
    "perform_wavefront_fitting",
    "plot_phase_error_profiles",
    "preprocess_phase_for_fitting",
    "find_wavefront_center",
    "fit_parabolic_phase",
    "fit_parabolic_phase_fast",
    "plot_phase_fit_results",
    # propagation
    "two_steps_fresnel_method",
    # utils
    "center_crop",
    "image_correction",
    "calculate_wavelength",
    "load_images",
    # zernike_analysis
    "analyze_and_visualize_zernike",
    "perform_zernike_analysis",
]
