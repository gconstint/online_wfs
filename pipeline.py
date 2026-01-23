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
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count

import numpy as np
from scipy.fft import fft2, fftshift

from core import (
    calculate_and_visualize_beam,
    analyze_focus_sampling_from_beam,
    preprocess_dpc,
    analyze_grating_data,
    calculate_harmonic_periods,
    calculate_magnification_correction,
    dpc_integration,
    perform_wavefront_fitting,
    two_steps_fresnel_method,
    center_crop,
    image_correction,
    load_images,
    analyze_and_visualize_zernike,
    # calibrate_focus_position,
    calculate_focus_from_dpc,
    calculate_undulator_source_distance,
    # analyze_mirror_surface,
    select_circular_roi,
    accurate_harmonic_periods,
)

# Constants
DEFAULT_CROP_SIZE = 2048
DEFAULT_LOWPASS_CUTOFF = 0.35
# MIN_MAGNIFICATION = 1.0
DEFAULT_ROTATION_ANGLE = 1.142  # Pre-computed rotation angle (degrees)


def print_separator(
    title: str = "", char: str = "=", width: int = 70, verbose: bool = True
) -> None:
    """Print a visual separator for workflow stages."""
    if not verbose:
        return
    if title:
        print(f"\n{char * width}")
        print(f"{title.center(width)}")
        print(f"{char * width}\n")
    else:
        print(f"\n{char * width}\n")


# =============================================================================
# Stage 1: Image Loading and Preprocessing
# =============================================================================


def load_and_preprocess_image(
    params: Dict[str, Any],
    verbose: bool = True,
    do_rotation: bool = True,
    rotation_angle: Optional[float] = None,
    img: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    crop_size: int = DEFAULT_CROP_SIZE,
) -> np.ndarray:
    """
    Stage 1: Load images and perform preprocessing.

    Loads raw, dark, and flat field images (or uses provided arrays),
    applies corrections, center-crops to standard size, and computes FFT.

    Parameters
    ----------
    params : dict
        Configuration parameters with pixel_size, pattern_period.
        If img/dark/flat are not provided, also requires image_path,
        dark_image_path, flat_image_path for file loading.
    verbose : bool
        Whether to print status messages
    do_rotation : bool
        Whether to perform image rotation correction to align peaks
        horizontally (default: True)
    rotation_angle : float, optional
        Pre-computed rotation angle in degrees. If provided and do_rotation=True,
        skips the expensive FFT + peak finding step and uses this angle directly.
        Set to 0.0 to skip rotation entirely. (default: None, compute from image)
    img : np.ndarray, optional
        Raw image data. If provided, skips file loading (for real-time analysis
        via EPICS or other control systems).
    dark : np.ndarray, optional
        Dark field image. If None and img is provided, no dark subtraction.
    flat : np.ndarray, optional
        Flat field image. If None and img is provided, no flat correction.
    crop_size : int, optional
        Target size for center cropping (default: 2048).

    Returns
    -------
    np.ndarray
        FFT of preprocessed image (img_fft)
    """
    print_separator("STAGE 1: Image Loading and Preprocessing", verbose=verbose)

    # Load images from file or use provided arrays
    if img is None:
        # File path mode: load from disk
        img, dark, flat = load_images(
            params["image_path"], params["dark_image_path"], params["flat_image_path"]
        )
    # else: Direct data mode - use provided img, dark, flat arrays

    # Apply dark field subtraction and flat field correction
    img = image_correction(img, flat=flat, dark=dark, epsilon=1e-8, normalize=False)

    # Center-crop to standard size
    img_cropped = center_crop(img, target_size=crop_size)

    # Calculate theoretical harmonic periods
    harmonic_periods = calculate_harmonic_periods(
        (img_cropped.shape[0], img_cropped.shape[1]),
        params["pixel_size"],
        params["pattern_period"],
    )
    params["harmonic_periods"] = harmonic_periods

    # Image rotation correction to align peaks horizontally
    if do_rotation:
        if rotation_angle is not None:
            # Use pre-computed rotation angle (skip expensive FFT + peak finding)
            angle = rotation_angle
        else:
            # Step 1: Compute initial FFT to find peak positions
            img32 = np.asarray(img_cropped, dtype=np.float32, order="C")
            img_fft_init = fftshift(fft2(img32, norm="ortho", workers=cpu_count()))

            # Step 2: Find accurate peak positions for 00, 01, 10 harmonics
            _, peak_positions = accurate_harmonic_periods(
                img_fft_init, harmonic_periods
            )

            # Step 3: Calculate rotation angle from peak positions
            peak_00 = peak_positions["00"]
            peak_01 = peak_positions["01"]
            peak_10 = peak_positions["10"]
            delta_y_h = peak_01[0] - peak_00[0]
            delta_x_h = peak_01[1] - peak_00[1]
            angle_h = np.arctan2(delta_y_h, delta_x_h) * 180 / np.pi
            delta_y_v = peak_10[0] - peak_00[0]
            delta_x_v = peak_10[1] - peak_00[1]
            angle_v = np.arctan2(delta_y_v, delta_x_v) * 180 / np.pi - 90
            angle = (angle_h + angle_v) / 2

        if verbose:
            print(f"Image rotation correction: {angle:.4f} degrees")

        # Apply rotation if angle is non-zero
        if abs(angle) > 1e-6:
            import cv2

            rows, cols = img_cropped.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            img_cropped_rotated = cv2.warpAffine(
                img_cropped, rotation_matrix, (cols, rows)
            )
        else:
            img_cropped_rotated = img_cropped

        # Compute FFT for frequency domain analysis (on rotated image)
        img32_rotated = np.asarray(img_cropped_rotated, dtype=np.float32, order="C")
        img_fft = fftshift(fft2(img32_rotated, norm="ortho", workers=cpu_count()))
    else:
        # Compute FFT directly without rotation
        img32 = np.asarray(img_cropped, dtype=np.float32, order="C")
        img_fft = fftshift(fft2(img32, norm="ortho", workers=cpu_count()))

    return img_fft


# =============================================================================
# Stage 2: Harmonic Extraction and DPC Calculation
# =============================================================================


def extract_harmonics_and_dpc(
    img_fft: np.ndarray,
    params: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Stage 2: Extract harmonics and compute DPC signals.

    Parameters
    ----------
    img_fft : np.ndarray
        FFT of the grating image
    params : dict
        Configuration parameters
    verbose : bool
        Whether to print status messages

    Returns
    -------
    dict
        Contains int00, int01, int10, dark_field01, dark_field10,
        dpc_x, dpc_y, virtual_pixel_size, updated params
    """
    print_separator("STAGE 2: Harmonic Extraction and DPC Calculation", verbose=verbose)

    results = analyze_grating_data(img_fft, None, params, plot_flag=False)
    (
        int00,
        int01,
        int10,
        dark_field01,
        dark_field10,
        dpc_x,
        dpc_y,
        virtual_pixel_size,
        params,
    ) = results

    return {
        "int00": int00,
        "int01": int01,
        "int10": int10,
        "dark_field01": dark_field01,
        "dark_field10": dark_field10,
        "dpc_x": dpc_x,
        "dpc_y": dpc_y,
        "virtual_pixel_size": virtual_pixel_size,
        "params": params,
    }


# =============================================================================
# Stage 3: Magnification Correction
# =============================================================================


def apply_magnification_correction(
    dpc_x: np.ndarray,
    dpc_y: np.ndarray,
    params: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Stage 3: Apply magnification correction to DPC signals.

    Parameters
    ----------
    dpc_x : np.ndarray
        Horizontal DPC signal
    dpc_y : np.ndarray
        Vertical DPC signal
    params : dict
        Configuration parameters
    verbose : bool
        Whether to print status messages

    Returns
    -------
    tuple
        (dpc_x_corrected, dpc_y_corrected, scale_factor)
    """
    print_separator("STAGE 3: Magnification Correction", verbose=verbose)

    scale_factor = calculate_magnification_correction(params)
    dpc_x_corrected = dpc_x * scale_factor
    dpc_y_corrected = dpc_y * scale_factor

    if verbose:
        print(f"Magnification correction factor: {scale_factor:.4f}")

    return dpc_x_corrected, dpc_y_corrected, scale_factor


# =============================================================================
# Stage 4: DPC Preprocessing and Phase Reconstruction
# =============================================================================


def reconstruct_phase(
    dpc_x: np.ndarray,
    dpc_y: np.ndarray,
    virtual_pixel_size: Tuple[float, float],
    verbose: bool = True,
    lowpass_cutoff: float = DEFAULT_LOWPASS_CUTOFF,
) -> np.ndarray:
    """
    Stage 4: Preprocess DPC and reconstruct phase using integration.

    Applies lowpass filtering, converts units, and integrates using
    Frankot-Chellappa method.

    Parameters
    ----------
    dpc_x : np.ndarray
        Corrected horizontal DPC signal
    dpc_y : np.ndarray
        Corrected vertical DPC signal
    virtual_pixel_size : tuple
        Effective pixel size (dy, dx) [m]
    verbose : bool
        Whether to print status messages
    lowpass_cutoff : float, optional
        Cutoff frequency for lowpass filter (default: 0.35).

    Returns
    -------
    np.ndarray
        Reconstructed phase [rad]
    """
    print_separator(
        "STAGE 4: DPC Preprocessing and Phase Reconstruction", verbose=verbose
    )

    # Apply lowpass filtering
    dpc_x_filtered = preprocess_dpc(dpc_x, lowpass_cutoff=lowpass_cutoff)
    dpc_y_filtered = preprocess_dpc(dpc_y, lowpass_cutoff=lowpass_cutoff)

    # Convert DPC from rad/pixel to rad/m for integration
    dpc_x_per_meter = dpc_x_filtered * virtual_pixel_size[1]
    dpc_y_per_meter = dpc_y_filtered * virtual_pixel_size[0]

    # Reconstruct phase using Frankot-Chellappa integration
    phase = dpc_integration(dpc_x_per_meter, dpc_y_per_meter)

    return phase


# =============================================================================
# Stage 5: Wavefront Fitting
# =============================================================================


def fit_wavefront(
    phase: np.ndarray,
    virtual_pixel_size: Tuple[float, float],
    wavelength: float,
    params: Dict[str, Any],
    verbose: bool = True,
    show_plots: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Stage 5: Fit wavefront with parabolic model.

    Parameters
    ----------
    phase : np.ndarray
        Reconstructed phase [rad]
    virtual_pixel_size : tuple
        Effective pixel size (dy, dx) [m]
    wavelength : float
        X-ray wavelength [m]
    params : dict
        System parameters, including total_dist for reference distance
    verbose : bool
        Whether to print status messages
    show_plots : bool
        Whether to show plots

    Returns
    -------
    tuple
        (fitted_phase, phase_error, fit_params)
    """
    print_separator("STAGE 5: Wavefront Fitting", verbose=verbose)

    fitted_phase, phase_error, fit_params = perform_wavefront_fitting(
        phase,
        virtual_pixel_size,
        wavelength,
        save_path=None,
        show_plots=show_plots,
        verbose=verbose,
    )

    # # Calculate focus position using DPC curvature method
    # reference_distance = params.get("total_dist", 0.465)
    # calculate_focus_from_dpc(
    #     fit_params=fit_params,
    #     wavelength=wavelength,
    #     reference_distance=reference_distance,
    #     verbose=verbose,
    # )

    return fitted_phase, phase_error, fit_params


# =============================================================================
# Stage 6: ROI Selection and Aberration Analysis
# =============================================================================


def analyze_aberrations(
    fitted_phase: np.ndarray,
    phase_error: np.ndarray,
    fit_params: list,
    params: Dict[str, Any],
    virtual_pixel_size: Tuple[float, float],
    interactive: bool = True,
    verbose: bool = True,
    show_plots: bool = True,
) -> Dict[str, Any]:
    """
    Stage 6: ROI selection and comprehensive aberration analysis.

    Includes focus calibration, source distance calculation, Zernike analysis,
    and mirror surface error analysis.

    Parameters
    ----------
    fitted_phase : np.ndarray
        Parabolic fitted phase [rad]
    phase_error : np.ndarray
        Phase error after parabolic fit [rad]
    fit_params : dict
        Wavefront fit parameters
    params : dict
        System parameters
    virtual_pixel_size : tuple
        Effective pixel size (dy, dx) [m]
    interactive : bool
        Whether to use interactive ROI selection
    verbose : bool
        Whether to print status messages
    show_plots : bool
        Whether to show plots

    Returns
    -------
    dict
        Contains roi_result, calibration_result, zernike_results,
        mirror_surface_results
    """
    print_separator("STAGE 6: ROI Selection and Aberration Analysis", verbose=verbose)

    # Select and crop ROI
    roi_result = select_circular_roi(
        phase_error,
        fit_params,
        virtual_pixel_size,
        params["wavelength"],
        interactive=interactive,
        save_path=None,
        verbose=verbose,
    )

    # # Focus Position Calibration (Zernike-based)
    # calibration_result = calibrate_focus_position(
    #     fitted_phase=fitted_phase,
    #     roi_result=roi_result,
    #     params=params,
    #     virtual_pixel_size=virtual_pixel_size,
    #     verbose=verbose,
    # )

    # Focus Position Calibration (DPC-based)
    calibration_result = calculate_focus_from_dpc(
        fit_params=fit_params,
        wavelength=params["wavelength"],
        reference_distance=params["total_dist"],
        verbose=verbose,
    )
    # Add 'R' key for compatibility with calculate_undulator_source_distance
    calibration_result["R"] = calibration_result["R_avg"]

    # Undulator Source Distance Calculation
    source_distance_result = calculate_undulator_source_distance(
        calibration_result=calibration_result,
        params=params,
        verbose=verbose,
    )
    if source_distance_result is not None:
        calibration_result["source_distance"] = source_distance_result

    # Zernike Analysis
    zernike_results = analyze_and_visualize_zernike(
        phase=roi_result["phase_error_cropped"],
        pixel_size=virtual_pixel_size,
        wavelength=params["wavelength"],
        num_terms=16,
        save_dir=None,
        save_path=None,
        show_plots=show_plots,
        verbose=verbose,
        aperture_center=roi_result["aperture_center"],
        aperture_radius_fraction=roi_result["aperture_radius_fraction"],
        use_radial_tukey_weight=True,
        tukey_alpha=0.3,
    )

    # # Mirror Surface Error Analysis
    # mirror_surface_results = analyze_mirror_surface(
    #     residual_phase=zernike_results["residual"],
    #     pixel_size=virtual_pixel_size,
    #     wavelength=params["wavelength"],
    #     grazing_angle_mrad=params.get("grazing_angle_mrad", 3.0),
    #     save_path=None,
    #     show_plots=show_plots,
    #     verbose=verbose,
    # )

    return {
        "roi_result": roi_result,
        "calibration_result": calibration_result,
        "zernike_results": zernike_results,
        # "mirror_surface_results": mirror_surface_results,
    }


# =============================================================================
# Stage 7: Beam Characterization at Detector Plane
# =============================================================================


def analyze_beam_at_detector(
    int00: np.ndarray,
    virtual_pixel_size: Tuple[float, float],
    verbose: bool = True,
    show_plots: bool = True,
) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """
    Stage 7: Analyze beam position and size at detector plane.

    Parameters
    ----------
    int00 : np.ndarray
        Zeroth-order intensity (transmission)
    virtual_pixel_size : tuple
        Effective pixel size (dy, dx) [m]
    verbose : bool
        Whether to print status messages
    show_plots : bool
        Whether to show plots

    Returns
    -------
    tuple
        (beam_position, beam_size)
    """
    print_separator("STAGE 7: Beam Characterization at Detector Plane", verbose=verbose)

    beam_position, beam_size = calculate_and_visualize_beam(
        int00,
        (virtual_pixel_size[0], virtual_pixel_size[1]),
        title="Self-Image Position and Size Analysis",
        show_plot=show_plots,
        verbose=verbose,
    )

    if verbose:
        print(f"Beam position: {beam_position}")
        print(f"Beam size (FWHM): {beam_size}")

    return beam_position, beam_size


# =============================================================================
# Stage 8: Focus Analysis via Back Propagation
# =============================================================================


def analyze_focus_by_propagation(
    int00: np.ndarray,
    phase: np.ndarray,
    virtual_pixel_size: Tuple[float, float],
    wavelength: float,
    propagation_distance: float,
    beam_size: Dict[str, float],
    verbose: bool = True,
    show_plots: bool = True,
) -> Dict[str, Any]:
    """
    Stage 8: Analyze focus by back-propagating to focus plane.

    Parameters
    ----------
    int00 : np.ndarray
        Zeroth-order intensity
    phase : np.ndarray
        Reconstructed phase [rad]
    virtual_pixel_size : tuple
        Effective pixel size (dy, dx) [m]
    wavelength : float
        X-ray wavelength [m]
    propagation_distance : float
        Distance to propagate (negative for back propagation) [m]
    beam_size : dict
        Beam size parameters from Stage 7
    verbose : bool
        Whether to print status messages
    show_plots : bool
        Whether to show plots

    Returns
    -------
    dict
        Contains focus_position, focus_size, focus_field, dx_focus, dy_focus
    """
    print_separator("STAGE 8: Focus Analysis via Back Propagation", verbose=verbose)

    dx, dy = virtual_pixel_size[1], virtual_pixel_size[0]

    # Determine required sampling at focus plane
    dx_focus, dy_focus, *_ = analyze_focus_sampling_from_beam(
        int00, dx, dy, wavelength, abs(propagation_distance), beam_size, verbose=verbose
    )

    # Calculate scale factors for proper sampling at focus
    scale_factor_x = dx / dx_focus
    scale_factor_y = dy / dy_focus
    if verbose:
        print(
            f"Focus sampling scale factors: x={scale_factor_x:.2f}, y={scale_factor_y:.2f}"
        )

    # Reconstruct complex field and propagate to focus
    # Note: use sqrt(int00) as amplitude since int00 is intensity
    complex_field = np.sqrt(int00) * np.exp(1j * phase)
    focus_field = two_steps_fresnel_method(
        complex_field,
        wavelength,
        -abs(propagation_distance),  # Negative for back propagation
        dx,
        dy,
        int(scale_factor_x),
        int(scale_factor_y),
    )

    # Analyze focus spot properties
    # Note: calculate_and_visualize_beam expects (py, px) i.e. (dy, dx) order
    # Use abs(focus_field)**2 for intensity since focus_field is complex amplitude
    focus_intensity = np.abs(focus_field) ** 2
    focus_position, focus_size = calculate_and_visualize_beam(
        focus_intensity,
        (dy_focus, dx_focus),
        title="Focus Position and Size Analysis",
        show_plot=show_plots,
        verbose=verbose,
    )

    if verbose:
        print(
            f"Focus position: {focus_position[0] * 1e9:.3f}nm, {focus_position[1] * 1e9:.3f}nm"
        )
        print(
            f"Focus size (FWHM): {focus_size['fwhm_x'] * 1e9:.3f}nm, {focus_size['fwhm_y'] * 1e9:.3f}nm"
        )

    return {
        "focus_position": focus_position,
        "focus_size": focus_size,
        "focus_field": focus_field,
        "dx_focus": dx_focus,
        "dy_focus": dy_focus,
    }


# =============================================================================
# Main Task Function (Generator-based Pipeline)
# =============================================================================


def task(
    params: dict,
    verbose: bool = True,
    show_plots: bool = True,
    do_rotation: bool = False,
    parallel: bool = True,
    img: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    rotation_angle: Optional[float] = None,
    lowpass_cutoff: float = DEFAULT_LOWPASS_CUTOFF,
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
    verbose : bool
        Whether to print status messages (default: True)
    show_plots : bool
        Whether to show plots (default: True)
    do_rotation : bool
        Whether to perform image rotation correction to align peaks
        horizontally (default: True)
    parallel : bool
        Whether to use parallel execution for independent paths (default: True).
        When True, Path 1 (wavefront + aberration) and Path 2 (focus analysis)
        run concurrently using ThreadPoolExecutor. Note: show_plots and
        interactive mode are automatically disabled in parallel mode.
    img : np.ndarray, optional
        Raw image data. If provided, skips file loading (for real-time analysis
        via EPICS or other control systems).
    dark : np.ndarray, optional
        Dark field image. If None and img is provided, no dark subtraction.
    flat : np.ndarray, optional
        Flat field image. If None and img is provided, no flat correction.
    crop_size : int, optional
        Target size for center cropping (default: 2048).
    rotation_angle : float, optional
        Pre-computed rotation angle in degrees. If provided and do_rotation=True,
        skips the expensive FFT + peak finding step.
    lowpass_cutoff : float, optional
        Cutoff frequency for lowpass filter in DPC preprocessing (default: 0.35).

    Yields
    ------
    tuple
        (checkpoint_name, results_dict) at each stage
    """
    # If parallel mode is enabled, disable show_plots for thread safety
    effective_show_plots = show_plots if not parallel else False
    # Stage 1: Image Loading and Preprocessing
    img_fft = load_and_preprocess_image(
        params,
        verbose=verbose,
        do_rotation=do_rotation,
        rotation_angle=rotation_angle,
        img=img,
        dark=dark,
        flat=flat,
        crop_size=crop_size,
    )

    # Stage 2: Harmonic Extraction and DPC Calculation
    harmonic_result = extract_harmonics_and_dpc(img_fft, params, verbose=verbose)
    int00 = harmonic_result["int00"]
    dpc_x = harmonic_result["dpc_x"]
    dpc_y = harmonic_result["dpc_y"]
    virtual_pixel_size = harmonic_result["virtual_pixel_size"]
    params = harmonic_result["params"]

    # Stage 3: Magnification Correction
    dpc_x_corrected, dpc_y_corrected, _ = apply_magnification_correction(
        dpc_x, dpc_y, params, verbose=verbose
    )

    # Stage 4: DPC Preprocessing and Phase Reconstruction
    phase = reconstruct_phase(
        dpc_x_corrected,
        dpc_y_corrected,
        virtual_pixel_size,
        verbose=verbose,
        lowpass_cutoff=lowpass_cutoff,
    )

    # =========================================================================
    # Parallel Processing: Two Independent Paths
    # =========================================================================
    # Path 1: fit_wavefront -> analyze_aberrations (wavefront & aberration analysis)
    # Path 2: analyze_beam_at_detector -> analyze_focus_by_propagation (focus analysis)
    # =========================================================================

    if parallel:
        # =====================================================================
        # PARALLEL MODE: Two Independent Paths using ThreadPoolExecutor
        # =====================================================================
        # Path 1: fit_wavefront -> analyze_aberrations
        # Path 2: analyze_beam_at_detector -> analyze_focus_by_propagation
        # =====================================================================

        def _path1_wavefront_aberration():
            """Path 1: Wavefront fitting and aberration analysis."""
            # Stage 5: Wavefront Fitting
            _fitted_phase, _phase_error, _fit_params = fit_wavefront(
                phase,
                virtual_pixel_size,
                params["wavelength"],
                params,
                verbose=verbose,
                show_plots=False,  # Disable plots in parallel execution
            )

            # Stage 6: ROI Selection and Aberration Analysis
            _aberration_result = analyze_aberrations(
                fitted_phase=_fitted_phase,
                phase_error=_phase_error,
                fit_params=_fit_params,
                params=params,
                virtual_pixel_size=virtual_pixel_size,
                interactive=False,  # Disable interactive mode in parallel
                verbose=verbose,
                show_plots=False,  # Disable plots in parallel execution
            )

            return {
                "fitted_phase": _fitted_phase,
                "phase_error": _phase_error,
                "fit_params": _fit_params,
                "aberration_result": _aberration_result,
            }

        def _path2_focus_analysis():
            """Path 2: Beam characterization and focus analysis."""
            # Stage 7: Beam Characterization at Detector Plane
            _beam_position, _beam_size = analyze_beam_at_detector(
                int00,
                virtual_pixel_size,
                verbose=verbose,
                show_plots=False,  # Disable plots in parallel execution
            )

            # Stage 8: Focus Analysis via Back Propagation
            _focus_result = analyze_focus_by_propagation(
                int00=int00,
                phase=phase,
                virtual_pixel_size=virtual_pixel_size,
                wavelength=params["wavelength"],
                propagation_distance=params["total_dist"],
                beam_size=_beam_size,
                verbose=verbose,
                show_plots=False,  # Disable plots in parallel execution
            )

            return {
                "beam_position": _beam_position,
                "beam_size": _beam_size,
                "focus_result": _focus_result,
            }

        # Execute both paths in parallel using ThreadPoolExecutor
        print_separator("PARALLEL EXECUTION: Path 1 & Path 2", verbose=verbose)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_path1 = executor.submit(_path1_wavefront_aberration)
            future_path2 = executor.submit(_path2_focus_analysis)

            # Wait for both paths to complete and get results
            path1_result = future_path1.result()
            path2_result = future_path2.result()

        # Extract results from Path 1
        fitted_phase = path1_result["fitted_phase"]
        phase_error = path1_result["phase_error"]
        fit_params = path1_result["fit_params"]
        aberration_result = path1_result["aberration_result"]

        # Extract results from Path 2
        beam_position = path2_result["beam_position"]
        beam_size = path2_result["beam_size"]
        focus_result = path2_result["focus_result"]

    else:
        # =====================================================================
        # SERIAL MODE: Sequential execution (supports show_plots/interactive)
        # =====================================================================

        # Stage 5: Wavefront Fitting
        fitted_phase, phase_error, fit_params = fit_wavefront(
            phase,
            virtual_pixel_size,
            params["wavelength"],
            params,
            verbose=verbose,
            show_plots=effective_show_plots,
        )

        # Stage 6: ROI Selection and Aberration Analysis
        aberration_result = analyze_aberrations(
            fitted_phase=fitted_phase,
            phase_error=phase_error,
            fit_params=fit_params,
            params=params,
            virtual_pixel_size=virtual_pixel_size,
            interactive=effective_show_plots,  # Only interactive if showing plots
            verbose=verbose,
            show_plots=effective_show_plots,
        )

        # Stage 7: Beam Characterization at Detector Plane
        beam_position, beam_size = analyze_beam_at_detector(
            int00,
            virtual_pixel_size,
            verbose=verbose,
            show_plots=effective_show_plots,
        )

        # Stage 8: Focus Analysis via Back Propagation
        focus_result = analyze_focus_by_propagation(
            int00=int00,
            phase=phase,
            virtual_pixel_size=virtual_pixel_size,
            wavelength=params["wavelength"],
            propagation_distance=params["total_dist"],
            beam_size=beam_size,
            verbose=verbose,
            show_plots=effective_show_plots,
        )

    # Checkpoint 1: Wavefront fitting results (after Stage 5)
    yield (
        "checkpoint_wavefront",
        {
            "phase": phase,
            "fitted_phase": fitted_phase,
            "phase_error": phase_error,
            "fit_params": fit_params,
            "virtual_pixel_size": virtual_pixel_size,
            "wavelength": params["wavelength"],
        },
    )

    # Checkpoint 2: Aberration analysis results (after Stage 6)
    yield (
        "checkpoint_aberration",
        {
            "calibration_result": aberration_result["calibration_result"],
            "zernike_results": aberration_result["zernike_results"],
            "roi_result": aberration_result["roi_result"],
            "phase_error": phase_error,
            "wavelength": params["wavelength"],
            "virtual_pixel_size": virtual_pixel_size,
        },
    )

    if verbose:
        print(
            f"Focus position: {focus_result['focus_position'][0] * 1e9:.3f}nm, "
            f"{focus_result['focus_position'][1] * 1e9:.3f}nm"
        )
        print(
            f"Focus size (FWHM): {focus_result['focus_size']['fwhm_x'] * 1e9:.3f}nm, "
            f"{focus_result['focus_size']['fwhm_y'] * 1e9:.3f}nm"
        )

    # Checkpoint 3: Focus analysis results (after Stage 8)
    yield (
        "checkpoint_focus",
        {
            "beam_position": beam_position,
            "beam_size": beam_size,
            "focus_position": focus_result["focus_position"],
            "focus_size": focus_result["focus_size"],
            "focus_field": focus_result["focus_field"],
            "dx_focus": focus_result["dx_focus"],
            "dy_focus": focus_result["dy_focus"],
        },
    )

    print_separator("Analysis Completed", verbose=verbose)
