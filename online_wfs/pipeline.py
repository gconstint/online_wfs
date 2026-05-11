"""
XGI Wavefront Reconstruction Pipeline

This module implements the complete Shearing Grating Interferometry (SGI)
reconstruction pipeline for XFEL wavefront sensing technology. The workflow
processes grating images to reconstruct wavefront phase information and
analyze beam properties.

The pipeline includes:
1. Image preprocessing and grating analysis
2. Harmonic extraction and DPC calculation
3. Magnification correction
4. DPC preprocessing and phase reconstruction
5. Wavefront fitting and aberration analysis (ROI selection, Zernike)
6. Beam characterization at detector plane
7. Focus analysis via back propagation
"""

from typing import Dict, Any, Tuple, Optional
from operator import itemgetter
import numpy as np

from .core.load_image import load_and_preprocess_image
from .core.grating_analysis import extract_harmonics_and_dpc
from .core.phase_analysis import reconstruct_phase
from .core.dpc_preprocess import run_dpc_fitting
from .core.utils import calculate_magnification_correction
from .core.zernike_analysis import (
    perform_zernike_analysis,
    visualize_zernike_analysis,
)
from .core.roi_utils import select_circular_roi
from .core.beam_analysis import calc_beam_size, calc_focus_by_back_prop


def apply_magnification_correction(
    dpc_x: np.ndarray,
    dpc_y: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    dpc_x : np.ndarray
        Horizontal DPC signal
    dpc_y : np.ndarray
        Vertical DPC signal
    params : dict
        Configuration parameters

    Returns
    -------
    tuple
        (dpc_x_corrected, dpc_y_corrected)
    """
    scale_factor = calculate_magnification_correction(params)
    dpc_x_corrected = dpc_x * scale_factor
    dpc_y_corrected = dpc_y * scale_factor

    return dpc_x_corrected, dpc_y_corrected


def _build_roi_fit_params_from_dpc_fit(
    dpc_fit_params: Dict[str, Any],
    phase_error: np.ndarray,
) -> list[float]:
    """Convert DPC-fit metadata into the compact fit-parameter format used downstream."""

    finite_values = np.asarray(phase_error, dtype=float)[np.isfinite(phase_error)]
    amplitude = (
        float(0.5 * (np.nanmax(finite_values) - np.nanmin(finite_values)))
        if finite_values.size > 0
        else 0.0
    )

    return [
        float(dpc_fit_params.get("x0", 0.0)),
        float(dpc_fit_params.get("y0", 0.0)),
        float(dpc_fit_params.get("R_x", np.inf)),
        float(dpc_fit_params.get("R_y", np.inf)),
        amplitude,
    ]


def _build_zernike_output_payload(
    aberration_analysis: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a compact coefficient payload for realtime output consumers."""

    sorted_terms = sorted(
        aberration_analysis.values(),
        key=lambda term: term["osa_index"],
    )

    return {
        "coefficients_nm_by_index": {
            f"Z{term['osa_index']}": float(term["coefficient_nm"])
            for term in sorted_terms
        },
        "coefficient_table": [
            {
                "Index": f"Z{term['osa_index']}",
                "Coeff (nm)": float(term["coefficient_nm"]),
            }
            for term in sorted_terms
        ],
    }


def analyze_aberrations(
    phase_error: np.ndarray,
    fit_params: list,
    params: Dict[str, Any],
    virtual_pixel_size: Tuple[float, float],
    show_plots: bool = False,
) -> Dict[str, Any]:
    """
    ROI selection and aberration analysis.

    Includes ROI selection, Zernike analysis, mirror surface error, and PSD
    analysis. Focus calibration is orchestrated separately in ``workflow`` once
    the ROI has been established.

    Parameters
    ----------
    phase_error : np.ndarray
        Phase error after parabolic fit [rad]
    fit_params : list
        Wavefront fit parameters
    params : dict
        System parameters
    virtual_pixel_size : tuple
        Effective pixel size (dy, dx) [m]
    show_plots : bool
        Whether to display plots for offline inspection.

    Returns
    -------
    dict
        Contains full Zernike fit results plus a compact output payload.
    """

    zero_zernike_indices = [0, 1, 2, 3, 4, 5]

    # if zero_zernike_indices is not None:
    #     zero_zernike_indices = [int(j) for j in zero_zernike_indices]
    #     print(f"Projecting out Zernike modes before fitting: {zero_zernike_indices}")

    zernike_num_terms = int(params.get("zernike_num_terms", 36))

    # Select and crop ROI
    roi_result = select_circular_roi(
        phase_error,
        fit_params,
        virtual_pixel_size,
    )

    # Zernike Analysis
    (
        coefficients,
        fitted_phase,
        residual,
        rms_error,
        zernike_terms,
        aberration_analysis,
    ) = perform_zernike_analysis(
        phase=roi_result["phase_error_cropped"],
        pixel_size=virtual_pixel_size,
        wavelength=params["wavelength"],
        num_terms=zernike_num_terms,
        aperture_center=roi_result["aperture_center"],
        aperture_radius_fraction=roi_result["aperture_radius_fraction"],
        use_radial_tukey_weight=True,
        tukey_alpha=0.3,
        zero_zernike_indices=zero_zernike_indices,
    )

    if show_plots:
        visualize_zernike_analysis(
            phase=roi_result["phase_error_cropped"],
            fitted_phase=fitted_phase,
            residual=residual,
            aberration_analysis=aberration_analysis,
            wavelength=params["wavelength"],
            pixel_size=virtual_pixel_size,
            title="Zernike Aberration Analysis",
        )

    return {
        "coefficients": coefficients,
        "fitted_phase": fitted_phase,
        "residual": residual,
        "rms_error": float(rms_error),
        "rms_error_nm": float(rms_error * params["wavelength"] / (2 * np.pi) * 1e9),
        "zernike_terms": zernike_terms,
        "aberration_analysis": aberration_analysis,
        "output": _build_zernike_output_payload(aberration_analysis),
    }


# =============================================================================
# Main Task Function (Generator-based Pipeline)
# =============================================================================


def task(
    params: dict,
    img: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    show_plots: bool = False,  # default to False
    do_rotation: bool = False,
    rotation_angle: Optional[float] = None,
):
    """
    Execute the complete XGI wavefront reconstruction pipeline.

    This is a generator-based pipeline that yields intermediate results
    at key checkpoints for online monitoring.

    Parameters
    ----------
    params : dict
        Configuration parameters dictionary containing:
        - image_path, dark_image_path, flat_image_path
        - pixel_size, pattern_period, wavelength
        - total_dist, source_dist
        - Optional: grazing_angle_mrad, crl_position, etc.
    show_plots : bool
        Whether to show plots for offline inspection (default: False)
    do_rotation : bool
        Whether to perform image rotation correction to align peaks
        horizontally (default: False)
    img : np.ndarray, optional
        Raw image data. If provided, skips file loading (for real-time analysis
        via EPICS or other control systems).
    dark : np.ndarray, optional
        Dark field image. If None and img is provided, no dark subtraction.
    flat : np.ndarray, optional
        Flat field image. If None and img is provided, no flat correction.
    rotation_angle : float, optional
        Pre-computed rotation angle in degrees. If provided and do_rotation=True,
        skips the expensive FFT + peak finding step.

    Yields
    ------
    tuple
        (checkpoint_name, results_dict) for each emitted stage
    """

    if "rotation_angle" in params:
        rotation_angle = params["rotation_angle"]
    if "do_rotation" in params:
        do_rotation = params["do_rotation"]

    img_fft = load_and_preprocess_image(
        params,
        do_rotation=do_rotation,
        rotation_angle=rotation_angle,
        img=img,
        dark=dark,
        flat=flat,
        # crop_size=crop_size,
    )

    # Stage 2: Harmonic Extraction and DPC Calculation
    harmonic_result = extract_harmonics_and_dpc(img_fft, params)
    (
        int00,
        dpc_x,
        dpc_y,
        virtual_pixel_size,
        params,
    ) = itemgetter(
        "int00",
        "dpc_x",
        "dpc_y",
        "virtual_pixel_size",
        "params",
    )(harmonic_result)

    dpc_x, dpc_y = apply_magnification_correction(
        dpc_x,
        dpc_y,
        params,
    )

    # dpc fitting
    dpc_fit_result = run_dpc_fitting(
        dpc_x=dpc_x,
        dpc_y=dpc_y,
        pixel_size=virtual_pixel_size,
        wavelength=params["wavelength"],
        use_robust=True,
        weight_sigma=0.4,
    )
    dpc_fit_params = dpc_fit_result["fit_params"]
    dpc_residual = dpc_fit_result["dpc_residual"]
    dpc_x_residual = np.asarray(dpc_residual["dpc_x_residual"], dtype=float)
    dpc_y_residual = np.asarray(dpc_residual["dpc_y_residual"], dtype=float)

    # print("R_x from DPC fit (m):", dpc_fit_params.get("R_x", np.inf))
    # print("R_y from DPC fit (m):", dpc_fit_params.get("R_y", np.inf))
    # output point1: focus distances
    yield (
        "focus_distances",
        {
            "R_x_m": float(dpc_fit_params.get("R_x", np.inf)),
            "R_y_m": float(dpc_fit_params.get("R_y", np.inf)),
        },
    )

    py, px = virtual_pixel_size
    phase_error = reconstruct_phase(
        dpc_x_residual * px,
        dpc_y_residual * py,
    )

    # beam analysis
    _, beam_size = calc_beam_size(
        int00,
        (virtual_pixel_size[0], virtual_pixel_size[1]),
    )
    # print("Calculated beam size (m):", beam_size)
    focus_size, *_ = calc_focus_by_back_prop(
        amplitude=phase_error,
        dx=virtual_pixel_size[1],
        dy=virtual_pixel_size[0],
        wavelength=params["wavelength"],
        beam_size=beam_size,
        propagation_distance_x=dpc_fit_params.get("R_x", None),
        propagation_distance_y=dpc_fit_params.get("R_y", None),
    )

    yield (
        "beam_analysis",
        {
            "beam_size": beam_size,
            "focus_size": focus_size,
            "phase_error": phase_error,
        },
    )

    # zernike analysis
    fit_params = _build_roi_fit_params_from_dpc_fit(dpc_fit_params, phase_error)

    zernike_results = analyze_aberrations(
        phase_error=phase_error,
        fit_params=fit_params,
        params=params,
        virtual_pixel_size=virtual_pixel_size,
        show_plots=show_plots,
    )
    # output point2: aberration analysis results
    yield "aberration_analysis", zernike_results["output"]

    # print("Estimated focus size (m):", focus_size)
