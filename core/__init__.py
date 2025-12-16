"""
Core analysis modules for the XGI wavefront sensor pipeline.
"""

from .beam_analysis import (
    calculate_and_visualize_beam,
    analyze_focus_sampling_from_beam,
    plot_beam_visualization,
)
from .dpc_preprocess import preprocess_dpc
from .grating_analysis import (
    analyze_grating_data,
    accurate_harmonic_periods,
    rotate_image_by_peaks,
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
    visualize_zernike_analysis,
)
from .focus_calibration import (
    calibrate_focus_position,
    calculate_focus_from_dpc,
    calculate_astigmatic_focus,
)
from .source_distance import calculate_undulator_source_distance
from .mirror_surface_analysis import analyze_mirror_surface
from .roi_utils import select_circular_roi

__all__ = [
    # beam_analysis
    "calculate_and_visualize_beam",
    "analyze_focus_sampling_from_beam",
    "plot_beam_visualization",
    # dpc_preprocess
    "preprocess_dpc",
    # grating_analysis
    "analyze_grating_data",
    "accurate_harmonic_periods",
    "rotate_image_by_peaks",
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
    "visualize_zernike_analysis",
    # focus_calibration
    "calibrate_focus_position",
    "calculate_focus_from_dpc",
    "calculate_astigmatic_focus",
    # source_distance
    "calculate_undulator_source_distance",
    # mirror_surface_analysis
    "analyze_mirror_surface",
    # roi_utils
    "select_circular_roi",
]
