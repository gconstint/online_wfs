"""
Focus Position Calibration Module

This module provides functions for calibrating the detector-to-focus distance
using Zernike defocus coefficient (C4) from wavefront analysis.

The calibration is based on the physical relationship between wavefront curvature
and the distance from detector to focus point.
"""

import numpy as np
from typing import Dict, Any, Union, Sequence

from .zernike_analysis import perform_zernike_analysis


def calibrate_focus_position(
    fitted_phase: np.ndarray,
    roi_result: Dict[str, Any],
    params: Dict[str, Any],
    virtual_pixel_size: Union[list, Sequence[float]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calibrate focus position based on Zernike coefficients of fitted_phase.

    Performs Zernike fitting of first 6 terms (j=0 to j=5) on fitted_phase to obtain:
    - C3 (Astigmatism 0°): Oblique astigmatism
    - C4 (Defocus): Defocus
    - C5 (Astigmatism 45°): Orthogonal astigmatism

    Calculates standard focus position (based on C4) and astigmatic focus positions (based on C3/C4/C5).

    Parameters
    ----------
    fitted_phase : np.ndarray
        Phase after paraboloid fitting [rad]
    roi_result : dict
        ROI selection result containing crop_info, aperture_center, aperture_radius_fraction
    params : dict
        System parameters containing wavelength, total_dist
    virtual_pixel_size : list
        Virtual pixel size (py, px) [m]
    verbose : bool
        Whether to print detailed results

    Returns
    -------
    dict
        Calibration result containing:
        - R, Delta_z, C4, C4_opd (standard focus calibration)
        - astigmatic_focus: astigmatic focus results (R_x, R_y, Delta_z_x, Delta_z_y, etc.)
    """
    # Get crop bounds from roi_result
    crop_info = roi_result["crop_info"]
    fitted_phase_cropped = fitted_phase[
        crop_info["row_start"] : crop_info["row_end"],
        crop_info["col_start"] : crop_info["col_end"],
    ]

    # Perform Zernike analysis on fitted_phase (need j=0 to j=5, 6 terms total)
    fitted_zernike_coeffs, _, _, _, _, _ = perform_zernike_analysis(
        phase=fitted_phase_cropped,
        pixel_size=virtual_pixel_size,
        wavelength=params["wavelength"],
        num_terms=6,  # j=0 to j=5, to get C3, C4, C5
        aperture_center=roi_result["aperture_center"],
        aperture_radius_fraction=roi_result["aperture_radius_fraction"],
        use_radial_tukey_weight=True,
        verbose=verbose,
    )

    # Get Zernike coefficients (only need C4 and C5)
    C4 = fitted_zernike_coeffs[4]  # Defocus (OSA j=4)
    C5 = fitted_zernike_coeffs[5]  # Astigmatism X-Y (OSA j=5)

    if verbose:
        print("\nFitted phase Zernike coefficients:")
        print(f"  C₄ (Defocus):     {C4:.6f} rad ({C4 / (2 * np.pi):.6f} λ)")
        print(f"  C₅ (Astig X-Y):   {C5:.6f} rad ({C5 / (2 * np.pi):.6f} λ)")

    # R0: Ideal detector-to-focus distance
    R0 = params["total_dist"]

    # r_max: Beam radius, using the physical radius from ROI selection
    cropped_size = min(roi_result["phase_error_cropped"].shape)
    r_max = (
        (cropped_size / 2)
        * roi_result["aperture_radius_fraction"]
        * np.mean(virtual_pixel_size)
    )

    # 1. Standard focus position calibration (based on C4 only)
    focus_result = calculate_focus_distance(
        C4=C4,
        R0=R0,
        r_max=r_max,
        wavelength=params["wavelength"],
        verbose=verbose,
    )

    # 2. Astigmatic focus position calibration (based on C4/C5)
    astigmatic_focus_result = calculate_astigmatic_focus(
        C4=C4,
        C5=C5,
        R0=R0,
        r_max=r_max,
        wavelength=params["wavelength"],
        verbose=verbose,
    )

    # Merge results
    result = focus_result.copy()
    result["astigmatic_focus"] = astigmatic_focus_result
    result["C5"] = C5

    return result


def calculate_focus_from_dpc(
    fit_params: list,
    wavelength: float,
    reference_distance: float = 0.465,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate focus position from DPC paraboloid fitting parameters (curvature superposition method).

    Physical Principle
    ------------------
    Convert distance to curvature (K = 1/R) for addition/subtraction, since curvatures are linearly additive:
        K_total = K_ideal + K_residual

    Expanded as distance formula:
        1/R_real = 1/R_0 + λ*a/π

    Where:
        - R_real: Actual focus position
        - R_0: Ideal reference distance
        - λ: X-ray wavelength
        - a: Fitted quadratic coefficient (unit: rad/m²)

    Quadratic coefficient a is extracted from fit parameters:
        a_x = A / Rx²
        a_y = A / Ry²

    Parameters
    ----------
    fit_params : list
        DPC paraboloid fit parameters [x0, y0, Rx, Ry, A]
        - x0, y0: Paraboloid center position [m]
        - Rx, Ry: X/Y direction virtual radius [m]
        - A: Amplitude [rad]
    wavelength : float
        X-ray wavelength [m]
    reference_distance : float
        Ideal reference distance R_0 [m], default 0.465 m
    verbose : bool
        Whether to print detailed results

    Returns
    -------
    dict
        Dictionary containing calibration results:
        - R_x : float - X-direction focus distance [m]
        - R_y : float - Y-direction focus distance [m]
        - R_avg : float - Average focus distance (R_x + R_y) / 2 [m]
        - Delta_x : float - X-direction focus offset R_x - R_0 [m]
        - Delta_y : float - Y-direction focus offset R_y - R_0 [m]
        - Delta_avg : float - Average focus offset [m]
        - a_x : float - X-direction quadratic coefficient [rad/m²]
        - a_y : float - Y-direction quadratic coefficient [rad/m²]
        - R_0 : float - Reference distance [m]
    """
    x0, y0, Rx, Ry, A = fit_params
    R_0 = reference_distance

    # Check validity
    if A == 0 or Rx <= 0 or Ry <= 0:
        if verbose:
            print(
                "Warning: Invalid fit parameters (A=0 or R<=0), cannot calculate focus."
            )
        return {
            "R_x": np.inf,
            "R_y": np.inf,
            "R_avg": np.inf,
            "Delta_x": np.nan,
            "Delta_y": np.nan,
            "Delta_avg": np.nan,
            "a_x": 0.0,
            "a_y": 0.0,
            "R_0": R_0,
        }

    # Extract quadratic coefficient from fit parameters
    a_x = A / (Rx**2)  # rad/m²
    a_y = A / (Ry**2)  # rad/m²

    # Residual wavefront curvature contribution
    K_residual_x = wavelength * a_x / np.pi  # 1/m
    K_residual_y = wavelength * a_y / np.pi  # 1/m

    # Total curvature = ideal curvature + residual curvature
    K_ideal = 1.0 / R_0  # 1/m
    K_total_x = K_ideal + K_residual_x
    K_total_y = K_ideal + K_residual_y

    # Convert back to distance: R_real = 1 / K_total
    R_x = 1.0 / K_total_x if abs(K_total_x) > 1e-14 else np.inf
    R_y = 1.0 / K_total_y if abs(K_total_y) > 1e-14 else np.inf

    # Focus offset
    Delta_x = R_x - R_0
    Delta_y = R_y - R_0

    # Average focus position and offset
    R_avg = (R_x + R_y) / 2.0
    Delta_avg = (Delta_x + Delta_y) / 2.0

    if verbose:
        print("\n" + "=" * 60)
        print("DPC Fit Focus Calibration Results (Curvature Superposition)".center(60))
        print("=" * 60)
        print("Formula: 1/R_real = 1/R_0 + λ*a/π")
        print("-" * 60)
        print("Input parameters:")
        print(f"  - Reference distance R_0:  {R_0:.6f} m ({R_0 * 1e3:.3f} mm)")
        print(f"  - Wavelength λ:            {wavelength * 1e9:.4f} nm")
        print(f"  - Fit amplitude A:         {A:.4f} rad")
        print(f"  - Virtual radius Rx:       {Rx * 1e6:.2f} μm")
        print(f"  - Virtual radius Ry:       {Ry * 1e6:.2f} μm")
        print("-" * 60)
        print("Quadratic coefficient a = A/R²:")
        print(f"  - a_x = {a_x:.4e} rad/m²")
        print(f"  - a_y = {a_y:.4e} rad/m²")
        print("-" * 60)
        print("Calibration results:")
        print(f"  - X-direction focus distance R_x: {R_x:.6f} m ({R_x * 1e3:.3f} mm)")
        print(f"  - Y-direction focus distance R_y: {R_y:.6f} m ({R_y * 1e3:.3f} mm)")
        print(
            f"  - Average focus distance R_avg:   {R_avg:.6f} m ({R_avg * 1e3:.3f} mm)"
        )
        print(f"  - X-direction offset ΔR_x:        {Delta_x * 1e3:.4f} mm")
        print(f"  - Y-direction offset ΔR_y:        {Delta_y * 1e3:.4f} mm")
        print(f"  - Average offset ΔR_avg:          {Delta_avg * 1e3:.4f} mm")
        print("=" * 60 + "\n")

    return {
        "R_x": R_x,
        "R_y": R_y,
        "R_avg": R_avg,
        "Delta_x": Delta_x,
        "Delta_y": Delta_y,
        "Delta_avg": Delta_avg,
        "a_x": a_x,
        "a_y": a_y,
        "R_0": R_0,
    }


def calculate_focus_distance(
    C4: float,
    R0: float,
    r_max: float,
    wavelength: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate actual detector-to-focus distance based on Zernike defocus coefficient C4.

    Physical Principle
    ------------------
    The curvature difference between ideal and actual wavefront can be approximated as (OPD units, meters):
        W_err(r) ≈ (r²/2R) - (r²/2R₀) = (r²/2) × (1/R - 1/R₀)

    Relationship between phase and OPD: φ = (2π/λ) × W

    Relationship between Zernike defocus coefficient and physical sag:
        Sag = √3 × C₄_opd

    Note: Although Z₄ = √3(2ρ² - 1) has ρ² coefficient of 2√3, the "2" here is for
    orthonormalization of Zernike polynomials, not representing twice the physical curvature.
    Physical sag should use √3 × C₄.

    Matching the physical formula:
        √3 × C₄_opd = r_max² / (2R) - r_max² / (2R₀) = (r_max²/2) × (1/R - 1/R₀)

    Solving:
        R = (1/R₀ + 2√3 × C₄_opd / r_max²)⁻¹

    Parameters
    ----------
    C4 : float
        Zernike defocus coefficient (OSA/ANSI index j=4), unit: rad.
    R0 : float
        Set ideal detector-to-focus distance [m].
    r_max : float
        Beam radius at detector (pupil diameter) [m].
    wavelength : float
        Wavelength [m].
    verbose : bool, optional
        Whether to print detailed calculation results. Default is True.

    Returns
    -------
    dict
        Dictionary containing calibration results:
        - R : float - Actual detector-to-focus distance [m]
        - Delta_z : float - Focus offset Δz = R - R₀ [m]
        - Delta_z_approx : float - Linear approximation offset [m]
        - C4 : float - Input Zernike defocus coefficient [rad]
        - C4_opd : float - C4 converted to OPD [m]

    Notes
    -----
    Sign convention (focus position calibration):
    - If Δz < 0 (R < R₀): Focus is too close to detector → focus needs to move away from detector
    - If Δz > 0 (R > R₀): Focus is too far from detector → focus needs to move toward detector
    """
    sqrt3 = np.sqrt(3)

    # Convert C4 from radians to OPD (meters)
    C4_opd = C4 * wavelength / (2 * np.pi)

    # Calculate full formula: R = (1/R₀ + 2√3 × C₄_opd / r_max²)⁻¹
    curvature_correction = 2 * sqrt3 * C4_opd / (r_max**2)

    # Handle extreme case to avoid division by zero
    inv_R = 1.0 / R0 + curvature_correction
    if abs(inv_R) < 1e-12:
        R = np.inf if inv_R >= 0 else -np.inf
    else:
        R = 1.0 / inv_R

    # Exact position error
    Delta_z = R - R0

    # Linear approximation: Δz ≈ -2√3 × R₀² × C₄_opd / r_max²
    Delta_z_approx = -2 * sqrt3 * (R0**2) * C4_opd / (r_max**2)

    if verbose:
        print("\n" + "=" * 60)
        print("Focus Position Calibration Results".center(60))
        print("=" * 60)
        print("Input parameters:")
        print(f"  - Zernike C₄ (phase):      {C4:.6f} rad ({C4 / (2 * np.pi):.6f} λ)")
        print(f"  - Zernike C₄ (OPD):        {C4_opd * 1e9:.6f} nm")
        print(f"  - Ideal distance R₀:       {R0:.6f} m ({R0 * 1e3:.3f} mm)")
        print(f"  - Beam radius r_max:       {r_max * 1e3:.3f} mm")
        print(f"  - Wavelength λ:            {wavelength * 1e9:.4f} nm")
        print("-" * 60)
        print("Calibration results:")
        print(f"  - Actual detector-to-focus distance R: {R:.6f} m ({R * 1e3:.3f} mm)")
        print(f"  - Focus offset Δz = R - R₀:            {Delta_z * 1e3:.6f} mm")
        print("=" * 60 + "\n")

    return {
        "R": R,
        "Delta_z": Delta_z,
        "Delta_z_approx": Delta_z_approx,
        "C4": C4,
        "C4_opd": C4_opd,
    }


def calculate_astigmatic_focus(
    C4: float,
    C5: float,
    R0: float,
    r_max: float,
    wavelength: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate X/Y direction focus positions for astigmatic case.

    When wavefront has astigmatism, X and Y directions have different radii of curvature,
    leading to two different focus positions. This function calculates independent focus
    positions for both directions based on Zernike coefficients C4 (Defocus) and C5 (Astigmatism).

    Physical Principle
    ------------------
    For OSA/ANSI normalized Zernike polynomials:
    - Z₄ (n=2, m=0):  √3 × (2ρ² - 1) → Defocus (average curvature)
    - Z₅ (n=2, m=2):  √6 × ρ² × cos(2θ) ∝ (x² - y²) → Orthogonal astigmatism (X-Y curvature difference)

    X/Y direction radii of curvature:
        1/R_x = 1/R₀ + (2√3 × C₄_opd + 2√6 × C₅_opd) / r_max²
        1/R_y = 1/R₀ + (2√3 × C₄_opd - 2√6 × C₅_opd) / r_max²

    Parameters
    ----------
    C4 : float
        Zernike defocus coefficient (OSA j=4, Defocus) [rad]
    C5 : float
        Zernike orthogonal astigmatism coefficient (OSA j=5, Astigmatism X-Y) [rad]
    R0 : float
        Set ideal detector-to-focus distance [m]
    r_max : float
        Beam radius at detector (pupil diameter) [m]
    wavelength : float
        Wavelength [m]
    verbose : bool
        Whether to print detailed results

    Returns
    -------
    dict
        Contains the following keys:
        - R_x : float - X-direction radius of curvature [m]
        - R_y : float - Y-direction radius of curvature [m]
        - Delta_z_x : float - X-direction focus offset [m]
        - Delta_z_y : float - Y-direction focus offset [m]
        - astigmatism_distance : float - Astigmatism distance |R_x - R_y| [m]
        - C4, C5 : float - Input Zernike coefficients [rad]
    """
    sqrt3 = np.sqrt(3)
    sqrt6 = np.sqrt(6)

    # Convert Zernike coefficients from radians to OPD (meters)
    C4_opd = C4 * wavelength / (2 * np.pi)
    C5_opd = C5 * wavelength / (2 * np.pi)

    inv_r_max_sq = 1.0 / (r_max**2)

    # Defocus contribution (same for both directions)
    defocus_correction = 2 * sqrt3 * C4_opd * inv_r_max_sq

    # Astigmatism contribution (opposite for both directions)
    # C5 (cos 2θ ∝ x² - y²) component corresponds to X-Y difference
    astig_correction = 2 * sqrt6 * C5_opd * inv_r_max_sq

    # X-direction curvature
    inv_R_x = 1.0 / R0 + defocus_correction + astig_correction
    if abs(inv_R_x) < 1e-12:
        R_x = np.inf if inv_R_x >= 0 else -np.inf
    else:
        R_x = 1.0 / inv_R_x

    # Y-direction curvature
    inv_R_y = 1.0 / R0 + defocus_correction - astig_correction
    if abs(inv_R_y) < 1e-12:
        R_y = np.inf if inv_R_y >= 0 else -np.inf
    else:
        R_y = 1.0 / inv_R_y

    # Focus offset
    Delta_z_x = R_x - R0
    Delta_z_y = R_y - R0

    # Astigmatism distance
    if np.isfinite(R_x) and np.isfinite(R_y):
        astigmatism_distance = abs(R_x - R_y)
    else:
        astigmatism_distance = np.inf

    if verbose:
        print("\n" + "=" * 70)
        print(
            "Astigmatic Focus Calibration Results (Based on Zernike C4/C5)".center(70)
        )
        print("=" * 70)
        print("Input Zernike coefficients:")
        print(f"  - C₄ (Defocus):      {C4:.6f} rad ({C4 / (2 * np.pi):.6f} λ)")
        print(f"  - C₅ (Astig X-Y):    {C5:.6f} rad ({C5 / (2 * np.pi):.6f} λ)")
        print(f"  - Ideal distance R₀: {R0:.6f} m ({R0 * 1e3:.3f} mm)")
        print(f"  - Beam radius r_max: {r_max * 1e3:.3f} mm")
        print("-" * 70)
        print("X direction (horizontal):")
        print(f"  - Radius of curvature R_x: {R_x:.6f} m ({R_x * 1e3:.3f} mm)")
        print(f"  - Focus offset Δz_x:       {Delta_z_x * 1e3:.6f} mm")
        print("Y direction (vertical):")
        print(f"  - Radius of curvature R_y: {R_y:.6f} m ({R_y * 1e3:.3f} mm)")
        print(f"  - Focus offset Δz_y:       {Delta_z_y * 1e3:.6f} mm")
        print("-" * 70)
        print(f"Astigmatism distance |Rx - Ry|: {astigmatism_distance * 1e3:.6f} mm")
        if astigmatism_distance < 1e-6:
            print("  → Approximately spherical wavefront, no significant astigmatism")
        else:
            print("  → ⚠ Astigmatism present, X/Y direction focus positions differ")
        print("=" * 70 + "\n")

    return {
        "R_x": R_x,
        "R_y": R_y,
        "Delta_z_x": Delta_z_x,
        "Delta_z_y": Delta_z_y,
        "astigmatism_distance": astigmatism_distance,
        "C4": C4,
        "C5": C5,
        "C4_opd": C4_opd,
        "C5_opd": C5_opd,
    }
