"""
Mirror Surface Error Analysis Module

This module provides functions for calculating mirror surface height error
and slope error from wavefront phase residuals, particularly relevant for
X-ray grazing incidence mirrors.

Key Functions:
    - calculate_mirror_surface_error: Calculate height and slope errors
    - visualize_mirror_surface_error: Visualize error distributions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Sequence


def calculate_mirror_surface_error(
    residual_phase: np.ndarray,
    pixel_size: Sequence[float],
    wavelength: float,
    grazing_angle_mrad: float = 3.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calculate mirror surface height error and slope error from phase residual.

    This function converts wavefront phase residual to mirror surface errors,
    accounting for the grazing incidence geometry typical of X-ray mirrors.

    Physical Relationships:
    - Wavefront error W = phase * λ / (2π)
    - Surface height error d = W / (2 * sin(θ)) for grazing incidence
    - Slope error: Mirror slope = Wavefront slope / 2 (INDEPENDENT of θ!)
      This is because wavefront slope = 2 × mirror slope for reflection.

    Parameters
    ----------
    residual_phase : np.ndarray
        2D phase residual array in radians (e.g., from Zernike fitting).
        NaN values indicate invalid regions.
    pixel_size : tuple of (float, float)
        Pixel size in meters as (dy, dx).
    wavelength : float
        X-ray wavelength in meters.
    grazing_angle_mrad : float, optional
        Grazing incidence angle in milliradians. Default is 3.0 mrad,
        a typical value for X-ray focusing mirrors.
        Typical ranges:
        - Soft X-ray mirrors: 5-10 mrad
        - Hard X-ray mirrors: 2-5 mrad
        - KB focusing mirrors: 2-4 mrad
    verbose : bool, optional
        If True, print analysis results. Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - surface_height_error_m : np.ndarray
            2D surface height error map in meters
        - height_rms_nm : float
            RMS of surface height error in nm
        - height_pv_nm : float
            Peak-to-valley of surface height error in nm
        - slope_x_urad : np.ndarray
            2D slope error in X direction in µrad
        - slope_y_urad : np.ndarray
            2D slope error in Y direction in µrad
        - slope_magnitude_urad : np.ndarray
            2D total slope error magnitude in µrad
        - slope_x_rms_urad : float
            RMS of X-direction slope error in µrad
        - slope_y_rms_urad : float
            RMS of Y-direction slope error in µrad
        - slope_rms_urad : float
            RMS of total slope error in µrad
        - grazing_angle_mrad : float
            Grazing angle used in calculation (mrad)
        - grazing_angle_deg : float
            Grazing angle in degrees

    Notes
    -----
    For X-ray mirrors at grazing incidence angle θ:
    - Wavefront error W relates to surface height error d by: W = 2d·sin(θ)
    - At very small angles, sin(θ) ≈ θ (in radians)
    - Lower grazing angles amplify the surface error sensitivity

    Examples
    --------
    >>> residual = zernike_results["residual"]  # from Zernike fitting
    >>> pixel_size = (10e-6, 10e-6)  # 10 µm pixels
    >>> wavelength = 1.55e-10  # 0.155 nm
    >>> results = calculate_mirror_surface_error(
    ...     residual, pixel_size, wavelength, grazing_angle_mrad=3.0
    ... )
    >>> print(f"Height RMS: {results['height_rms_nm']:.3f} nm")
    >>> print(f"Slope RMS: {results['slope_rms_urad']:.3f} µrad")
    """
    # Extract pixel sizes (in wavefront coordinates)
    dy, dx = pixel_size[0], pixel_size[1]

    # Convert grazing angle to degrees and calculate sin(θ)
    grazing_angle_deg = grazing_angle_mrad * 1e-3 * 180 / np.pi
    sin_theta = np.sin(grazing_angle_mrad * 1e-3)  # θ in radians

    # ========== Surface Height Error Calculation ==========
    # Convert wavefront phase to height error: phase (rad) -> height (m)
    # phase = 2π * W / λ, so W = phase * λ / (2π)
    # Surface height error: d = W / (2 * sin(θ))
    wavefront_error_m = residual_phase * wavelength / (2 * np.pi)  # in meters
    surface_height_error_m = wavefront_error_m / (2 * sin_theta)  # in meters

    # Calculate height error statistics (ignore NaN values)
    valid_height = surface_height_error_m[~np.isnan(surface_height_error_m)]
    height_rms_nm = np.std(valid_height) * 1e9  # RMS in nm
    height_pv_nm = (np.max(valid_height) - np.min(valid_height)) * 1e9  # PV in nm

    # ========== Slope Error Calculation ==========
    # IMPORTANT: Calculate slope directly from wavefront phase gradient
    # This avoids the grazing geometry coordinate stretch issue.
    #
    # Physical relationships:
    # - Wavefront slope = dW/dx = (dφ/dx) * λ/(2π)  [rad, i.e., m/m]
    # - For reflection: Wavefront slope = 2 × Mirror slope
    # - Therefore: Mirror slope = Wavefront slope / 2
    #
    # This is INDEPENDENT of grazing angle! The grazing angle affects
    # height error, but mirror slope is simply half of wavefront slope.

    # Calculate wavefront phase gradient (in wavefront coordinates)
    phase_grad_y, phase_grad_x = np.gradient(residual_phase, dy, dx)  # rad/m

    # Convert to wavefront slope: dW/dx = (dφ/dx) * λ/(2π)
    wavefront_slope_x = (
        phase_grad_x * wavelength / (2 * np.pi)
    )  # dimensionless (m/m = rad)
    wavefront_slope_y = phase_grad_y * wavelength / (2 * np.pi)

    # Mirror slope = Wavefront slope / 2 (for reflection geometry)
    mirror_slope_x = wavefront_slope_x / 2  # rad
    mirror_slope_y = wavefront_slope_y / 2  # rad

    # Convert to µrad (microradians) for conventional representation
    slope_x_urad = mirror_slope_x * 1e6  # µrad
    slope_y_urad = mirror_slope_y * 1e6  # µrad

    # Calculate total slope error magnitude
    slope_magnitude = np.sqrt(mirror_slope_x**2 + mirror_slope_y**2)
    slope_magnitude_urad = slope_magnitude * 1e6

    # Calculate slope error statistics (ignore NaN values)
    valid_slope_x = slope_x_urad[~np.isnan(slope_x_urad)]
    valid_slope_y = slope_y_urad[~np.isnan(slope_y_urad)]

    # RMS = sqrt(mean(x²)) for error analysis (deviation from zero)
    # For X and Y components, std is appropriate since mean should be ~0 (random errors)
    slope_x_rms_urad = np.sqrt(np.mean(valid_slope_x**2))
    slope_y_rms_urad = np.sqrt(np.mean(valid_slope_y**2))

    # Total RMS: vector sum of component RMS values
    # slope_rms = sqrt(slope_x_rms² + slope_y_rms²)
    # This is correct for uncorrelated X and Y errors
    slope_rms_urad = np.sqrt(slope_x_rms_urad**2 + slope_y_rms_urad**2)

    # Print results if verbose
    if verbose:
        print("Mirror Surface Error Analysis (from Zernike Residual)")
        print("-" * 70)
        print(
            f"Grazing angle: {grazing_angle_mrad:.1f} mrad "
            f"({grazing_angle_deg:.3f}°, sin(θ) = {sin_theta:.6f})"
        )
        print()
        print("Surface Height Error:")
        print(f"  RMS:  {height_rms_nm:.3f} nm")
        print(f"  PV:   {height_pv_nm:.3f} nm")
        print()
        print("Slope Error (Mirror Surface):")
        print(f"  X-direction RMS: {slope_x_rms_urad:.3f} µrad")
        print(f"  Y-direction RMS: {slope_y_rms_urad:.3f} µrad")
        print(f"  Total RMS:       {slope_rms_urad:.3f} µrad")
        print("=" * 70)

    # Return results dictionary
    return {
        "surface_height_error_m": surface_height_error_m,
        "height_rms_nm": height_rms_nm,
        "height_pv_nm": height_pv_nm,
        "slope_x_urad": slope_x_urad,
        "slope_y_urad": slope_y_urad,
        "slope_magnitude_urad": slope_magnitude_urad,
        "slope_x_rms_urad": slope_x_rms_urad,
        "slope_y_rms_urad": slope_y_rms_urad,
        "slope_rms_urad": slope_rms_urad,
        "grazing_angle_mrad": grazing_angle_mrad,
        "grazing_angle_deg": grazing_angle_deg,
    }


def visualize_mirror_surface_error(
    results: Dict[str, Any],
    pixel_size: Sequence[float],
    title: str = "Mirror Surface Error Analysis",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Visualize mirror surface height error and slope error distributions.

    Creates a 2x2 figure showing:
    - Surface height error map
    - Slope error magnitude map
    - X-direction slope error map
    - Y-direction slope error map

    Parameters
    ----------
    results : dict
        Results dictionary from calculate_mirror_surface_error().
    pixel_size : tuple of (float, float)
        Pixel size in meters as (dy, dx).
    title : str, optional
        Figure title. Default is "Mirror Surface Error Analysis".
    save_path : str, optional
        Directory path to save the figure. If None, figure is not saved.
    show_plot : bool, optional
        If True, display the plot. Default is True.
    """
    # Extract data from results
    height_error = results["surface_height_error_m"] * 1e9  # Convert to nm
    slope_x = results["slope_x_urad"]
    slope_y = results["slope_y_urad"]
    slope_mag = results["slope_magnitude_urad"]

    # Calculate extent for proper axis labels
    ny, nx = height_error.shape
    dy, dx = pixel_size
    extent = [
        -nx * dx * 1e3 / 2,
        nx * dx * 1e3 / 2,
        -ny * dy * 1e3 / 2,
        ny * dy * 1e3 / 2,
    ]  # in mm

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color limits using percentiles for better visualization
    h_vmin, h_vmax = np.nanpercentile(height_error, [2, 98])
    s_vmax = np.nanpercentile(slope_mag, 98)
    sx_lim = np.nanpercentile(np.abs(slope_x), 98)
    sy_lim = np.nanpercentile(np.abs(slope_y), 98)

    # Surface height error
    ax1 = axes[0, 0]
    im1 = ax1.imshow(
        height_error,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=h_vmin,
        vmax=h_vmax,
    )
    ax1.set_title(
        f"Surface Height Error\n"
        f"RMS: {results['height_rms_nm']:.3f} nm, PV: {results['height_pv_nm']:.3f} nm"
    )
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    plt.colorbar(im1, ax=ax1, label="Height Error (nm)")

    # Slope magnitude
    ax2 = axes[0, 1]
    im2 = ax2.imshow(
        slope_mag,
        extent=extent,
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=s_vmax,
    )
    ax2.set_title(f"Slope Error Magnitude\nRMS: {results['slope_rms_urad']:.3f} µrad")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    plt.colorbar(im2, ax=ax2, label="Slope Error (µrad)")

    # X-direction slope error
    ax3 = axes[1, 0]
    im3 = ax3.imshow(
        slope_x,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=-sx_lim,
        vmax=sx_lim,
    )
    ax3.set_title(
        f"X-Direction Slope Error\nRMS: {results['slope_x_rms_urad']:.3f} µrad"
    )
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    plt.colorbar(im3, ax=ax3, label="Slope Error (µrad)")

    # Y-direction slope error
    ax4 = axes[1, 1]
    im4 = ax4.imshow(
        slope_y,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=-sy_lim,
        vmax=sy_lim,
    )
    ax4.set_title(
        f"Y-Direction Slope Error\nRMS: {results['slope_y_rms_urad']:.3f} µrad"
    )
    ax4.set_xlabel("X (mm)")
    ax4.set_ylabel("Y (mm)")
    plt.colorbar(im4, ax=ax4, label="Slope Error (µrad)")

    # Add grazing angle info to main title
    grazing_info = (
        f"Grazing angle: {results['grazing_angle_mrad']:.1f} mrad "
        f"({results['grazing_angle_deg']:.3f}°)"
    )
    fig.suptitle(f"{title}\n{grazing_info}", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        img_path = os.path.join(save_path, "mirror_surface_error.png")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"Mirror surface error figure saved to: {img_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_mirror_surface(
    residual_phase: np.ndarray,
    pixel_size: Sequence[float],
    wavelength: float,
    grazing_angle_mrad: float = 3.0,
    save_path: Optional[str] = None,
    show_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete mirror surface error analysis with optional visualization.

    This is a high-level wrapper function that performs surface error
    calculation and visualization in one call.

    Parameters
    ----------
    residual_phase : np.ndarray
        2D phase residual array in radians (e.g., from Zernike fitting).
    pixel_size : tuple of (float, float)
        Pixel size in meters as (dy, dx).
    wavelength : float
        X-ray wavelength in meters.
    grazing_angle_mrad : float, optional
        Grazing incidence angle in milliradians. Default is 3.0 mrad.
    save_path : str, optional
        Directory path to save visualization. If None, figure is not saved.
    show_plots : bool, optional
        If True, display visualization plots. Default is True.
    verbose : bool, optional
        If True, print analysis results. Default is True.

    Returns
    -------
    dict
        Complete analysis results from calculate_mirror_surface_error().

    Examples
    --------
    >>> results = analyze_mirror_surface(
    ...     residual_phase=zernike_results["residual"],
    ...     pixel_size=virtual_pixel_size,
    ...     wavelength=params["wavelength"],
    ...     grazing_angle_mrad=params.get("grazing_angle_mrad", 3.0),
    ...     save_path=save_path,
    ...     show_plots=True,
    ... )
    """
    # Calculate surface errors
    results = calculate_mirror_surface_error(
        residual_phase=residual_phase,
        pixel_size=pixel_size,
        wavelength=wavelength,
        grazing_angle_mrad=grazing_angle_mrad,
        verbose=verbose,
    )

    # Visualize results
    if show_plots:
        visualize_mirror_surface_error(
            results=results,
            pixel_size=pixel_size,
            save_path=save_path,
            show_plot=True,
        )

    return results
