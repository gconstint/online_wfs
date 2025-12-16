"""
Beam analysis module for X-ray grating interferometry.

This module provides functions for analyzing beam properties including:
- Beam position and size calculation
- FWHM and 1/e² radius measurements
- Focus sampling rate estimation
- Gaussian beam propagation analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from typing import Tuple, Dict, Optional


# Physical constants
FWHM_TO_W0_FACTOR = np.sqrt(2 * np.log(2))  # FWHM = w0 * sqrt(2*ln2) ≈ 1.177
SIGMA_TO_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))  # FWHM = 2.355 * sigma


def _gaussian_normalized(
    x: np.ndarray, amplitude: float, mean: float, sigma: float, offset: float
) -> np.ndarray:
    """
    Normalized Gaussian function for fitting.

    Args:
        x: Input coordinates
        amplitude: Peak amplitude
        mean: Center position
        sigma: Standard deviation
        offset: Background offset

    Returns:
        Gaussian profile values
    """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2)) + offset


def _estimate_background(profile: np.ndarray, edge_fraction: float = 0.1) -> float:
    """
    Estimate background level from profile edges.

    Args:
        profile: 1D intensity profile
        edge_fraction: Fraction of data to use from edges (default 10%)

    Returns:
        Estimated background level
    """
    edge_pixel_count = max(1, int(len(profile) * edge_fraction))
    edge_values = np.concatenate(
        [profile[:edge_pixel_count], profile[-edge_pixel_count:]]
    )
    return np.median(edge_values)


def _normalize_profile(profile: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize profile with background subtraction.

    Args:
        profile: 1D intensity profile

    Returns:
        Tuple of (normalized_profile, background_level)
    """
    background = _estimate_background(profile)
    profile_bg_sub = profile - background
    profile_max = np.max(profile_bg_sub)

    if profile_max > 0:
        normalized = profile_bg_sub / profile_max
    else:
        normalized = profile_bg_sub.copy()

    return normalized, background


def _fit_gaussian_profile(
    profile: np.ndarray, coords: np.ndarray, verbose: bool = True
) -> Tuple[float, float, Optional[tuple]]:
    """
    Fit Gaussian to 1D profile and calculate FWHM and w0.

    Args:
        profile: 1D intensity profile
        coords: Physical coordinates (in microns)
        verbose: Whether to print status messages (default: True)

    Returns:
        Tuple of (fwhm, w0, fit_params) where fit_params is None if fitting fails
    """
    # Normalize profile
    normalized_profile, _ = _normalize_profile(profile)

    # Find peak position
    norm_max_idx = np.argmax(normalized_profile)
    data_range = abs(coords[-1] - coords[0])

    try:
        # Initial parameter estimation
        amplitude_init = 1.0
        mean_init = coords[norm_max_idx]

        # Better sigma initialization: estimate from half-width at half-max
        half_max = 0.5 * normalized_profile[norm_max_idx]
        above_half = np.where(normalized_profile >= half_max)[0]
        if len(above_half) > 1:
            # Estimate FWHM from data, then convert to sigma
            estimated_fwhm = abs(coords[above_half[-1]] - coords[above_half[0]])
            sigma_init = estimated_fwhm / SIGMA_TO_FWHM_FACTOR
        else:
            sigma_init = data_range / 8

        offset_init = 0.0

        # Define bounds to avoid non-physical results
        bounds = (
            [0.5, coords[0] - data_range * 0.5, data_range * 0.001, -0.2],
            [1.5, coords[-1] + data_range * 0.5, data_range * 0.8, 0.2],
        )

        # Perform curve fitting with reduced iterations (faster convergence with good init)
        popt, _ = curve_fit(
            _gaussian_normalized,
            coords,
            normalized_profile,
            p0=[amplitude_init, mean_init, sigma_init, offset_init],
            bounds=bounds,
            maxfev=2000,  # Reduced from 10000 - usually converges in < 100
        )

        amplitude, mean, sigma, offset = popt

        # Only calculate R² when verbose (saves computation)
        if verbose:
            residuals = normalized_profile - _gaussian_normalized(coords, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((normalized_profile - np.mean(normalized_profile)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"Fitting R²: {r_squared:.3f}")

        # Calculate FWHM from sigma: FWHM = 2.355 * sigma
        fwhm = SIGMA_TO_FWHM_FACTOR * abs(sigma)

        # Convert FWHM to 1/e² radius: w0 = FWHM / sqrt(2*ln2)
        w0 = fwhm / FWHM_TO_W0_FACTOR

        return fwhm, w0, popt

    except Exception as e:
        if verbose:
            print(f"Gaussian fitting failed: {e}")

        # Fallback: simple FWHM calculation
        target_value = np.max(normalized_profile) * 0.5
        indices = np.where(normalized_profile >= target_value)[0]

        if len(indices) > 1:
            fwhm = abs(coords[indices[-1]] - coords[indices[0]])
        else:
            fwhm = abs(coords[-1] - coords[0]) * 0.1

        w0 = fwhm / FWHM_TO_W0_FACTOR

        return fwhm, w0, None


def calculate_and_visualize_beam(
    intensity: np.ndarray,
    virtual_pixel_size: Tuple[float, float],
    title: str = "Beam Analysis",
    save_path=None,
    show_plot: bool = True,
    verbose: bool = True,
) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """
    Calculate beam position and size from 2D intensity distribution.

    This function analyzes a beam by:
    1. Extracting horizontal and vertical intensity profiles
    2. Fitting Gaussian functions to each profile
    3. Calculating beam center position and FWHM
    4. Visualizing the results with overlays

    Physical Model:
    ---------------
    Assumes Gaussian beam profile: I(x,y) = I₀ · exp(-2(x²+y²)/w₀²)
    where w₀ is the 1/e² beam radius.

    The Full Width at Half Maximum (FWHM) relates to w₀ as:
        FWHM = w₀ · sqrt(2·ln2) ≈ 1.177 · w₀

    Args:
        intensity: 2D intensity distribution array
        virtual_pixel_size: Pixel size as (py, px) tuple in meters
        title: Title for visualization plots
        save_path: Path to save visualization
        show_plot: Whether to show plots (default: True)
        verbose: Whether to print status messages (default: True)

    Returns:
        Tuple of (beam_position, beam_size) where:
        - beam_position: (x, y) beam centroid in meters
        - beam_size: Dict with 'fwhm_x' and 'fwhm_y' in meters
    """
    # =========================================================================
    # Step 1: Extract intensity profiles
    # =========================================================================
    height, width = intensity.shape

    # Average along each axis to get 1D profiles
    # Horizontal profile: average over all rows (axis=0)
    profile_horizontal = np.mean(intensity, axis=0)
    # Vertical profile: average over all columns (axis=1)
    profile_vertical = np.mean(intensity, axis=1)

    # =========================================================================
    # Step 2: Create physical coordinate axes
    # =========================================================================
    # Convert pixel coordinates to physical coordinates (microns for display)
    py, px = virtual_pixel_size  # Unpack for clarity

    # Coordinates centered at image center
    x_coords_um = (np.arange(width) - width / 2) * px * 1e6
    y_coords_um = (np.arange(height) - height / 2) * py * 1e6

    # =========================================================================
    # Step 3: Fit Gaussian profiles to extract beam parameters
    # =========================================================================
    # Fit horizontal direction (X)
    fwhm_x_um, w0_x_um, fit_params_x = _fit_gaussian_profile(
        profile_horizontal, x_coords_um, verbose=verbose
    )

    # Fit vertical direction (Y)
    fwhm_y_um, w0_y_um, fit_params_y = _fit_gaussian_profile(
        profile_vertical, y_coords_um, verbose=verbose
    )

    # =========================================================================
    # Step 4: Extract beam center position from fit parameters
    # =========================================================================
    # Gaussian fit parameters: [amplitude, mean, sigma, offset]
    # The 'mean' parameter gives the beam center position

    if fit_params_x is not None:
        _, beam_center_x_um, _, _ = fit_params_x
    else:
        beam_center_x_um = 0.0
        if verbose:
            print("Warning: Horizontal Gaussian fit failed, using image center (x=0)")

    if fit_params_y is not None:
        _, beam_center_y_um, _, _ = fit_params_y
    else:
        beam_center_y_um = 0.0
        if verbose:
            print("Warning: Vertical Gaussian fit failed, using image center (y=0)")

    # =========================================================================
    # Step 5: Display results
    # =========================================================================
    if verbose:
        print(f"Beam center: ({beam_center_x_um:.3f}, {beam_center_y_um:.3f}) µm")
        print(f"FWHM: x = {fwhm_x_um:.3f} µm, y = {fwhm_y_um:.3f} µm")
        print(f"w0 (1/e² radius): x = {w0_x_um:.3f} µm, y = {w0_y_um:.3f} µm")

    # =========================================================================
    # Step 6: Visualize beam analysis results
    # =========================================================================
    if show_plot:
        plot_beam_visualization(
            intensity,
            virtual_pixel_size,
            beam_center_x_um,
            beam_center_y_um,
            fwhm_x_um,
            fwhm_y_um,
            fit_params_x,
            fit_params_y,
            title,
            save_path,
        )

    # =========================================================================
    # Step 7: Return results in SI units (meters)
    # =========================================================================
    beam_position = (
        beam_center_x_um * 1e-6,  # Convert µm to m
        beam_center_y_um * 1e-6,
    )

    beam_size = {
        "fwhm_x": fwhm_x_um * 1e-6,  # Convert µm to m
        "fwhm_y": fwhm_y_um * 1e-6,
    }

    return beam_position, beam_size


def plot_beam_visualization(
    intensity: np.ndarray,
    virtual_pixel_size: Tuple[float, float],
    beam_x_um: float,
    beam_y_um: float,
    fwhm_x: float,
    fwhm_y: float,
    fit_params_x: Optional[tuple],
    fit_params_y: Optional[tuple],
    title: str = "Beam Analysis",
    save_path=None,
) -> None:
    """
    Plot intensity map with beam analysis overlays.

    Creates a 2x2 subplot showing:
    - Intensity map with beam center and FWHM ellipse
    - Horizontal profile with Gaussian fit
    - Vertical profile with Gaussian fit
    - 3D surface plot

    Args:
        intensity: 2D intensity array
        virtual_pixel_size: Pixel size [px_x, px_y] in meters
        beam_x_um: Beam center x-coordinate in microns
        beam_y_um: Beam center y-coordinate in microns
        fwhm_x: FWHM in x-direction (microns)
        fwhm_y: FWHM in y-direction (microns)
        fit_params_x: Gaussian fit parameters for x-profile (or None)
        fit_params_y: Gaussian fit parameters for y-profile (or None)
        title: Plot title
    """
    height, width = intensity.shape

    # Calculate physical coordinates
    x_size = width * virtual_pixel_size[0]
    y_size = height * virtual_pixel_size[1]
    x_coords = (np.arange(width) - width / 2) * virtual_pixel_size[0] * 1e6
    y_coords = (np.arange(height) - height / 2) * virtual_pixel_size[1] * 1e6

    # Calculate profiles
    horizontal_profile = np.mean(intensity, axis=0)
    vertical_profile = np.mean(intensity, axis=1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    extent = [-x_size / 2 * 1e6, x_size / 2 * 1e6, -y_size / 2 * 1e6, y_size / 2 * 1e6]

    # ===== Subplot 1: Intensity map =====
    im = axes[0, 0].imshow(intensity, cmap="viridis", extent=extent, origin="lower")
    plt.colorbar(im, ax=axes[0, 0], label="Intensity (a.u.)", fraction=0.046, pad=0.04)

    # Add beam center marker and crosshairs
    axes[0, 0].plot(beam_x_um, beam_y_um, "rx", markersize=10, label="Beam Center")
    axes[0, 0].axhline(y=beam_y_um, color="r", linestyle="--", alpha=0.5)
    axes[0, 0].axvline(x=beam_x_um, color="r", linestyle="--", alpha=0.5)

    # Add FWHM ellipse
    ellipse = plt.matplotlib.patches.Ellipse(
        (beam_x_um, beam_y_um),
        fwhm_x,
        fwhm_y,
        fill=False,
        color="r",
        linestyle="-",
        linewidth=2,
        label="Beam Size (FWHM)",
    )
    axes[0, 0].add_patch(ellipse)

    # Add info text box
    info_text = (
        f"Beam Center: ({beam_x_um:.3f}, {beam_y_um:.3f}) µm\n"
        f"FWHM X: {fwhm_x:.3f} µm\n"
        f"FWHM Y: {fwhm_y:.3f} µm"
    )
    axes[0, 0].text(
        0.02,
        0.98,
        info_text,
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    axes[0, 0].set_title(title)
    axes[0, 0].set_xlabel("x (µm)")
    axes[0, 0].set_ylabel("y (µm)")
    axes[0, 0].legend(loc="lower right")

    # ===== Subplot 2: Horizontal profile =====
    if fit_params_x is not None:
        amplitude_x, mean_x, sigma_x, offset_x = fit_params_x
        normalized_h, _ = _normalize_profile(horizontal_profile)
        fitted_curve_x = _gaussian_normalized(
            x_coords, amplitude_x, mean_x, sigma_x, offset_x
        )

        axes[1, 0].plot(
            x_coords, normalized_h, "bx", markersize=4, label="Data (norm.)"
        )
        axes[1, 0].plot(
            x_coords, fitted_curve_x, "r--", linewidth=2, label="Gaussian fit"
        )
    else:
        axes[1, 0].plot(x_coords, horizontal_profile, "bx", markersize=4, label="Data")

    axes[1, 0].set_title(f"Horizontal Profile (FWHM = {fwhm_x:.3f} µm)")
    axes[1, 0].set_xlabel("x (µm)")
    axes[1, 0].set_ylabel("Intensity (norm.)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ===== Subplot 3: Vertical profile =====
    if fit_params_y is not None:
        amplitude_y, mean_y, sigma_y, offset_y = fit_params_y
        normalized_v, _ = _normalize_profile(vertical_profile)
        fitted_curve_y = _gaussian_normalized(
            y_coords, amplitude_y, mean_y, sigma_y, offset_y
        )

        axes[0, 1].plot(
            y_coords, normalized_v, "bx", markersize=4, label="Data (norm.)"
        )
        axes[0, 1].plot(
            y_coords, fitted_curve_y, "r--", linewidth=2, label="Gaussian fit"
        )
    else:
        axes[0, 1].plot(y_coords, vertical_profile, "bx", markersize=4, label="Data")

    axes[0, 1].set_title(f"Vertical Profile (FWHM = {fwhm_y:.3f} µm)")
    axes[0, 1].set_xlabel("y (µm)")
    axes[0, 1].set_ylabel("Intensity (norm.)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ===== Subplot 4: 3D surface =====
    ax3d = fig.add_subplot(2, 2, 4, projection="3d")
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ax3d.plot_surface(x_mesh, y_mesh, intensity, cmap="viridis", alpha=0.8)
    ax3d.set_title("3D Intensity Profile")
    ax3d.set_xlabel("x (µm)")
    ax3d.set_ylabel("y (µm)")
    ax3d.set_zlabel("Intensity")

    plt.tight_layout()
    if save_path:
        img_path = os.path.join(save_path, f"{title.split(' ')[0]}_beam_analysis.png")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print("MESSAGE: The beam analysis image is saved")
        print("-" * 50)
    plt.show()


def analyze_focus_sampling_from_beam(
    amplitude: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    propagation_distance: float,
    beam_size: Dict[str, float],
    verbose: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate required focus sampling based on Gaussian beam propagation.

    Physical Model:
    ---------------
    Uses Gaussian beam propagation equation to back-calculate the beam waist
    at focus from the measured beam size at the detector plane:

        w(z)² = w₀² [1 + (z·λ/(π·w₀²))²]

    where:
        - w(z): 1/e² beam radius at distance z from focus
        - w₀: beam waist (minimum radius) at focus
        - z: propagation distance
        - λ: wavelength

    The required pixel size at focus is determined by two criteria:
    1. Magnification method: dx_focus = dx / M, where M = w(z) / w₀
    2. Angular resolution: dx_focus = (λ / N·dx) · |z|

    We use the smaller (more conservative) value to ensure proper sampling.

    Args:
        amplitude: Amplitude distribution at detector (used for array shape)
        dx: Pixel size in X direction at detector (m)
        dy: Pixel size in Y direction at detector (m)
        wavelength: X-ray wavelength (m)
        propagation_distance: Distance from focus to detector (m, positive)
        beam_size: Dict with 'fwhm_x' and 'fwhm_y' at detector (m)
        verbose: Whether to print status messages (default: True)

    Returns:
        Tuple of (dx_focus_x, dx_focus_y, divergence_x, divergence_y, w0_x, w0_y)
        where:
        - dx_focus_x, dx_focus_y: Required pixel sizes at focus (m)
        - divergence_x, divergence_y: Beam divergence half-angles (rad)
        - w0_x, w0_y: Beam waist radii at focus (m)
    """
    # Extract measured FWHM at detector
    fwhm_detector_x = beam_size["fwhm_x"]
    fwhm_detector_y = beam_size["fwhm_y"]

    if verbose:
        print(
            f"Current FWHM: fwhm_x={fwhm_detector_x * 1e6:.3f} μm, "
            f"fwhm_y={fwhm_detector_y * 1e6:.3f} μm"
        )

    # =========================================================================
    # Step 1: Convert FWHM to 1/e² radius at detector
    # =========================================================================
    # For Gaussian beam: FWHM = w₀ · sqrt(2·ln2) ≈ 1.177 · w₀
    w_detector_x = fwhm_detector_x / FWHM_TO_W0_FACTOR
    w_detector_y = fwhm_detector_y / FWHM_TO_W0_FACTOR

    # =========================================================================
    # Step 2: Back-propagate to find beam waist w₀ at focus
    # =========================================================================
    # Solve: w(z)² = w₀² [1 + (z·λ/(π·w₀²))²] for w₀
    # This is a quartic equation in w₀, solved numerically

    def gaussian_beam_equation(w0, w_z):
        """
        Gaussian beam propagation equation residual.

        Returns zero when w0 satisfies: w(z)² = w₀²[1 + (z·λ/(π·w₀²))²]
        """
        rayleigh_range = np.pi * w0**2 / wavelength
        propagation_factor = (propagation_distance / rayleigh_range) ** 2
        return w0**2 * (1 + propagation_factor) - w_z**2

    # Solve for beam waist in each direction
    # Initial guess: assume focus spot is ~100 nm (typical for X-ray focusing)
    w0_x = fsolve(lambda w0: gaussian_beam_equation(w0, w_detector_x), x0=100e-9)[0]
    w0_y = fsolve(lambda w0: gaussian_beam_equation(w0, w_detector_y), x0=100e-9)[0]

    # =========================================================================
    # Step 3: Calculate derived beam parameters
    # =========================================================================
    # FWHM at focus
    fwhm_focus_x = w0_x * FWHM_TO_W0_FACTOR
    fwhm_focus_y = w0_y * FWHM_TO_W0_FACTOR

    # Far-field divergence half-angle: θ = λ/(π·w₀)
    divergence_x = wavelength / (np.pi * w0_x)
    divergence_y = wavelength / (np.pi * w0_y)

    if verbose:
        print("Focus beam waist:")
        print(
            f"  X: fwhm={fwhm_focus_x * 1e6:.3f} μm, w0={w0_x * 1e6:.3f} μm, "
            f"divergence={divergence_x * 1e6:.3f} μrad"
        )
        print(
            f"  Y: fwhm={fwhm_focus_y * 1e6:.3f} μm, w0={w0_y * 1e6:.3f} μm, "
            f"divergence={divergence_y * 1e6:.3f} μrad"
        )

    # =========================================================================
    # Step 4: Calculate required pixel size at focus
    # =========================================================================

    # Method 1: Magnification-based sampling
    # ----------------------------------------
    # Physical principle: The beam expands as it propagates from focus
    # Magnification M = w(z) / w₀ (ratio of beam sizes at detector vs focus)
    # To maintain same sampling quality: dx_focus = dx_detector / M
    magnification_x = w_detector_x / w0_x
    magnification_y = w_detector_y / w0_y

    dx_mag = dx / magnification_x
    dy_mag = dy / magnification_y

    # Method 2: Angular resolution sampling (Nyquist criterion)
    # ----------------------------------------
    # Physical principle: Diffraction-limited angular resolution
    #
    # The detector array acts as a finite aperture with size L = N·dx
    # This limits the angular resolution to: Δθ ≈ λ/L = λ/(N·dx)
    #
    # When we back-propagate to focus at distance z, this angular
    # uncertainty translates to spatial uncertainty:
    #     Δx_focus = Δθ · z = (λ/(N·dx)) · z
    #
    # To properly sample the focus, we need: dx_focus ≤ Δx_focus

    N_x = amplitude.shape[1]  # Number of pixels in X direction
    N_y = amplitude.shape[0]  # Number of pixels in Y direction

    # Angular resolution (radians)
    angular_res_x = wavelength / (N_x * dx)
    angular_res_y = wavelength / (N_y * dy)

    # Spatial resolution at focus plane
    dx_ang = angular_res_x * abs(propagation_distance)
    dy_ang = angular_res_y * abs(propagation_distance)

    # =========================================================================
    # Display sampling analysis
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("FOCUS SAMPLING ANALYSIS")
        print("=" * 70)

        print("\nDetector parameters:")
        print(f"  Array size: {N_x} × {N_y} pixels")
        print(f"  Pixel size: dx={dx * 1e6:.3f} μm, dy={dy * 1e6:.3f} μm")
        print(f"  Field of view: {N_x * dx * 1e3:.3f} × {N_y * dy * 1e3:.3f} mm")

        print("\nBeam magnification:")
        print(
            f"  M_x = {magnification_x:.3f}× (beam expanded {magnification_x:.3f} times in X)"
        )
        print(
            f"  M_y = {magnification_y:.3f}× (beam expanded {magnification_y:.3f} times in Y)"
        )

        print("\nAngular resolution at detector:")
        print(f"  Δθ_x = {angular_res_x * 1e6:.3f} μrad (λ/L_x)")
        print(f"  Δθ_y = {angular_res_y * 1e6:.3f} μrad (λ/L_y)")

        print("\nRequired pixel size at focus:")
        print(
            f"  Method 1 (Magnification): dx={dx_mag * 1e9:.3f} nm, dy={dy_mag * 1e9:.3f} nm"
        )
        print(
            f"  Method 2 (Angular res.):  dx={dx_ang * 1e9:.3f} nm, dy={dy_ang * 1e9:.3f} nm"
        )

    # =========================================================================
    # Step 5: Choose conservative (finer) sampling
    # =========================================================================
    # Use the smaller pixel size to ensure we don't undersample
    dx_focus = min(dx_mag, dx_ang)
    dy_focus = min(dy_mag, dy_ang)

    if verbose:
        print("\nSelected (conservative):")
        print(f"  dx_focus = {dx_focus * 1e9:.3f} nm")
        print(f"  dy_focus = {dy_focus * 1e9:.3f} nm")
        print("=" * 70 + "\n")

    return dx_focus, dy_focus, divergence_x, divergence_y, w0_x, w0_y
