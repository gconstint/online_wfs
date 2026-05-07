# -*- coding: utf-8 -*-
"""
Zernike aberration analysis module using OSA/ANSI standard indexing.

This module provides functions for:
- Calculating Zernike polynomials using OSA/ANSI single-index notation
- Fitting phase data to Zernike basis
- Analyzing and identifying optical aberrations
"""

import json
import os

import numpy as np
from scipy.ndimage import shift as ndi_shift
from scipy.special import factorial


def osa_to_nm(j):
    """
    Convert OSA/ANSI standard index j to radial (n) and azimuthal (m) orders.

    The OSA/ANSI standard uses a single index j (0-indexed) that maps to
    unique (n, m) pairs following a specific ordering scheme.

    Parameters:
    -----------
    j : int
        OSA/ANSI standard index (0-indexed).

    Returns:
    --------
    tuple: (n, m)
        Radial order (n) and azimuthal order (m).

    Examples:
    ---------
    j=0: (0, 0) - Piston
    j=1: (1, -1) - Tilt Y
    j=2: (1, 1) - Tilt X
    j=3: (2, -2) - Astigmatism 0°
    j=4: (2, 0) - Defocus
    """
    if j < 0:
        raise ValueError("OSA index must be 0 or greater.")

    # Find radial order n
    n = 0
    while (n + 1) * (n + 2) / 2 <= j:
        n += 1

    # Calculate azimuthal order m
    m = 2 * j - n * (n + 2)

    return n, m


def nm_to_osa(n, m):
    """
    Convert radial (n) and azimuthal (m) orders to OSA/ANSI index j.

    Parameters:
    -----------
    n : int
        Radial order.
    m : int
        Azimuthal order.

    Returns:
    --------
    int: OSA/ANSI index j
    """
    if abs(m) > n or (n - abs(m)) % 2 != 0:
        raise ValueError(f"Invalid (n, m) = ({n}, {m}) for Zernike polynomial")

    j = (n * (n + 2) + m) // 2
    return j


def get_zernike_name(j):
    """
    Get the standard name for a Zernike polynomial by OSA/ANSI index.

    Parameters:
    -----------
    j : int
        OSA/ANSI index.

    Returns:
    --------
    str: Name of the aberration
    """
    # OSA/ANSI standard names (extended to j=35, 36 terms)
    names = {
        # Order 0 (j=0)
        0: "Piston",
        # Order 1 (j=1-2): Tilt
        1: "Tilt Y (Vertical)",
        2: "Tilt X (Horizontal)",
        # Order 2 (j=3-5): Defocus and Astigmatism
        3: "Astigmatism 0° (Oblique)",
        4: "Defocus",
        5: "Astigmatism 45° (Vertical)",
        # Order 3 (j=6-9): Coma and Trefoil
        6: "Trefoil Y",
        7: "Coma Y (Vertical)",
        8: "Coma X (Horizontal)",
        9: "Trefoil X",
        # Order 4 (j=10-14): Primary Spherical and higher
        10: "Quadrafoil Y",
        11: "Secondary Astigmatism 0°",
        12: "Primary Spherical",
        13: "Secondary Astigmatism 45°",
        14: "Quadrafoil X",
        # Order 5 (j=15-20): Secondary Coma and Pentafoil
        15: "Pentafoil Y",
        16: "Secondary Trefoil Y",
        17: "Secondary Coma Y",
        18: "Secondary Coma X",
        19: "Secondary Trefoil X",
        20: "Pentafoil X",
        # Order 6 (j=21-27): Secondary Spherical and higher
        21: "Hexafoil Y",
        22: "Secondary Quadrafoil Y",
        23: "Secondary Secondary Astigmatism 0°",
        24: "Secondary Spherical",
        25: "Secondary Secondary Astigmatism 45°",
        26: "Secondary Quadrafoil X",
        27: "Hexafoil X",
        # Order 7 (j=28-35): Tertiary aberrations
        28: "Heptafoil Y",
        29: "Secondary Pentafoil Y",
        30: "Tertiary Trefoil Y",
        31: "Tertiary Coma Y",
        32: "Tertiary Coma X",
        33: "Tertiary Trefoil X",
        34: "Secondary Pentafoil X",
        35: "Heptafoil X",
    }

    if j in names:
        return names[j]
    else:
        n, m = osa_to_nm(j)
        return f"Z{j} (n={n}, m={m})"


def calculate_zernike_polynomial(j, rho, theta):
    """
    Calculate a single Zernike polynomial using OSA/ANSI indexing with proper normalization.

    Parameters:
    -----------
    j : int
        OSA/ANSI standard index.
    rho : ndarray
        Normalized radial coordinates (0 to 1).
    theta : ndarray
        Angular coordinates (0 to 2π).

    Returns:
    --------
    ndarray: Normalized Zernike polynomial values
    """
    n, m = osa_to_nm(j)

    # Calculate radial component R_n^m(ρ)
    R = np.zeros_like(rho)
    if (n - abs(m)) % 2 == 0:
        for k in range((n - abs(m)) // 2 + 1):
            coef = (-1) ** k * factorial(n - k)
            coef /= (
                factorial(k)
                * factorial((n + abs(m)) // 2 - k)
                * factorial((n - abs(m)) // 2 - k)
            )
            R += coef * rho ** (n - 2 * k)

    # Apply angular component based on m
    if m > 0:  # Positive m: cosine term
        Z = R * np.cos(m * theta)
    elif m < 0:  # Negative m: sine term
        Z = R * np.sin(abs(m) * theta)
    else:  # m = 0: no angular dependence
        Z = R

    # Apply OSA/ANSI normalization factor
    # N = sqrt(2(n+1)) for m ≠ 0, sqrt(n+1) for m = 0
    if m != 0:
        norm_factor = np.sqrt(2 * (n + 1))
    else:
        norm_factor = np.sqrt(n + 1)

    Z_normalized = norm_factor * Z

    return Z_normalized


def fit_zernike_polynomials(phase, basis, indices, weights=None):
    """
    Fit phase data to Zernike basis using least squares.

    Parameters:
    -----------
    phase : ndarray
        2D phase map in radians.
    basis : list
        List of Zernike basis functions.
    indices : list
        List of corresponding OSA/ANSI indices.
    weights : ndarray, optional
        Weight map for weighted least squares fitting.

    Returns:
    --------
    tuple: (coefficients, fitted_phase, residual)
        - coefficients: Fitted Zernike coefficients in radians
        - fitted_phase: Reconstructed phase from Zernike fit
        - residual: Phase - fitted_phase
    """
    # Determine support mask from basis functions
    stacked_basis = np.stack(basis, axis=0)
    support_mask = np.sum(stacked_basis**2, axis=0) > 0

    # Also exclude NaN values from phase
    valid_data_mask = support_mask & ~np.isnan(phase)

    # Flatten arrays
    phase_flat = phase.flatten()
    mask_flat = valid_data_mask.flatten()

    # Construct design matrix
    basis_matrix = np.array([b.flatten()[mask_flat] for b in basis]).T
    y = phase_flat[mask_flat]

    # Check if we have enough valid data
    if len(y) < len(basis):
        raise ValueError(
            f"Not enough valid data points ({len(y)}) for {len(basis)} Zernike terms"
        )

    # Apply weights if provided
    if weights is not None:
        w = weights.flatten()[mask_flat]
        w = np.clip(w, 0.0, None)
        sqrt_w = np.sqrt(w)
        basis_matrix = basis_matrix * sqrt_w[:, None]
        y = y * sqrt_w

    # Least squares fit
    coefficients, *_ = np.linalg.lstsq(basis_matrix, y, rcond=None)

    # Reconstruct fitted phase - preserve NaN mask
    fitted_phase_full = np.sum(
        [coeff * basis[i] for i, coeff in enumerate(coefficients)], axis=0
    )

    # Apply the same valid_data_mask to fitted_phase
    fitted_phase = np.full_like(phase, np.nan)
    fitted_phase[valid_data_mask] = fitted_phase_full[valid_data_mask]

    # Calculate residual - will also have NaN where data is invalid
    residual = phase - fitted_phase

    # # Diagnostic: Check actual contribution of largest coefficient
    # if len(coefficients) > 0:
    #     max_idx = np.argmax(np.abs(coefficients))
    #     max_coeff = coefficients[max_idx]
    #     max_basis_contribution = max_coeff * basis[max_idx]
    #     print("\n  Zernike fitting diagnostics:")
    #     print(f"    Largest coefficient: Z{indices[max_idx]} = {max_coeff:.6f} rad")
    #     print(
    #         f"    Its basis function range: [{np.nanmin(basis[max_idx]):.6f}, {np.nanmax(basis[max_idx]):.6f}]"
    #     )
    #     print(
    #         f"    Its phase contribution range: [{np.nanmin(max_basis_contribution):.6f}, {np.nanmax(max_basis_contribution):.6f}] rad"
    #     )
    #     print(
    #         f"    Fitted phase total range: [{np.nanmin(fitted_phase):.6f}, {np.nanmax(fitted_phase):.6f}] rad"
    #     )

    return coefficients, fitted_phase, residual


def analyze_aberrations(coefficients, indices, wavelength=632.8e-9):
    """
    Analyze Zernike coefficients and identify aberrations.

    Parameters:
    -----------
    coefficients : ndarray
        Fitted Zernike coefficients in radians.
    indices : list
        List of OSA/ANSI indices corresponding to coefficients.
    wavelength : float
        Wavelength in meters.

    Returns:
    --------
    dict: Aberration analysis with detailed information for each term
    """
    aberration_analysis = {}

    for i, (coeff, j) in enumerate(zip(coefficients, indices)):
        n, m = osa_to_nm(j)
        aberration_name = get_zernike_name(j)

        # Convert coefficient to different units
        coefficient_rad = float(coeff)
        coefficient_waves = coefficient_rad / (2 * np.pi)
        coefficient_nm = coefficient_rad * wavelength / (2 * np.pi) * 1e9

        # Calculate RMS contribution
        # For OSA/ANSI normalized Zernike polynomials:
        # ∫∫ Z_j² dA = π (over unit disk of area π)
        # So variance = ∫∫ (c_j × Z_j)² dA / (π) = c_j² × π / π = c_j²
        # Therefore RMS contribution of term j = |c_j|
        rms_rad = abs(coeff)
        rms_nm = rms_rad * wavelength / (2 * np.pi) * 1e9

        aberration_analysis[f"Z{j}"] = {
            "osa_index": j,
            "name": aberration_name,
            "n": n,
            "m": m,
            "coefficient_rad": coefficient_rad,
            "coefficient_waves": float(coefficient_waves),
            "coefficient_nm": float(coefficient_nm),
            "rms_rad": float(rms_rad),
            "rms_nm": float(rms_nm),
        }

    return aberration_analysis


def perform_zernike_analysis(
    phase,
    pixel_size,
    wavelength,
    num_terms=36,
    save_dir=None,
    aperture_center=None,
    aperture_radius_fraction=1.0,
    use_radial_tukey_weight=True,
    tukey_alpha=0.5,
    zero_zernike_indices=None,
    phase_unit="radians",  # "radians" or "meters"
):
    """
    Perform comprehensive Zernike aberration analysis on phase data.

    Parameters:
    -----------
    phase : ndarray
        2D phase map.
    pixel_size : float or tuple
        Pixel size in meters, or (py, px) tuple.
    wavelength : float
        Wavelength in meters.
    num_terms : int
        Number of Zernike terms to fit (default: 36 for j=0 to j=35).
    save_dir : str, optional
        Directory to save analysis results.
    aperture_center : tuple, optional
        (y, x) center position in meters for aperture alignment.
    aperture_radius_fraction : float
        Fraction of image size to use as aperture radius (default: 1.0).
    use_radial_tukey_weight : bool
        Whether to use Tukey window weighting at edges (default: True).
    tukey_alpha : float
        Tukey window parameter (default: 0.5).
    zero_zernike_indices : list, optional
        List of OSA indices to project out before fitting.
    phase_unit : str
        Unit of the input phase map. 'radians' or 'meters'.
        If 'meters', it will be converted to radians using the wavelength.
        Default is 'radians'.

    Returns:
    --------
    tuple: (coefficients, fitted_phase, residual, rms_error, zernike_terms, aberration_analysis)
    """
    size_y, size_x = phase.shape
    size = min(size_y, size_x)

    # Handle unit conversion
    if phase_unit.lower() == "meters":
        print(
            f"Converting phase from meters to radians (wavelength={wavelength:.3e} m)"
        )
        phase = phase * 2 * np.pi / wavelength
    elif phase_unit.lower() == "radians":
        # Check if values look suspiciously small (like meters)
        phase_range = np.nanmax(phase) - np.nanmin(phase)
        if phase_range < 1e-4 and phase_range > 0:
            print(
                f"WARNING: Input phase range is very small ({phase_range:.3e}). "
                f"Are you sure the input is in radians? If it is in meters, set phase_unit='meters'."
            )
    else:
        raise ValueError(
            f"Unknown phase_unit: {phase_unit}. Must be 'radians' or 'meters'."
        )

    # Determine maximum OSA index needed
    j_max = num_terms - 1

    # Unify pixel size
    if isinstance(pixel_size, (float, int)):
        pixel_size_y = float(pixel_size)
        pixel_size_x = float(pixel_size)
    else:
        pixel_size_y, pixel_size_x = float(pixel_size[0]), float(pixel_size[1])

    # Center alignment if aperture center is provided
    if aperture_center is not None:
        cy_m, cx_m = aperture_center  # (y0, x0) in meters
        cy_px = size_y / 2.0 + cy_m / pixel_size_y
        cx_px = size_x / 2.0 + cx_m / pixel_size_x
        shift_y = size_y / 2.0 - cy_px
        shift_x = size_x / 2.0 - cx_px
        phase = ndi_shift(phase, shift=(shift_y, shift_x), order=1, mode="nearest")

    # Create coordinate grids
    yy, xx = np.mgrid[0:size_y, 0:size_x]
    cy = size_y / 2.0
    cx = size_x / 2.0
    radius = (min(size_y, size_x) / 2.0) * float(aperture_radius_fraction)
    rho = np.sqrt(((xx - cx) / radius) ** 2 + ((yy - cy) / radius) ** 2)
    aperture_mask = rho <= 1.0

    # Copy phase for processing
    phase_processed = phase.copy()

    # Generate Zernike basis directly on the analysis grid so the basis uses the
    # same aperture center and radius as the actual fit.
    theta = np.arctan2(yy - cy, xx - cx)

    basis = []
    indices = []
    for j in range(j_max + 1):
        Z = calculate_zernike_polynomial(j, rho, theta)
        # Set values outside unit circle to 0 (rho > 1 means outside aperture)
        Z[rho > 1.0] = 0
        basis.append(Z)
        indices.append(j)

    # Optional: edge weighting with Tukey window
    weights = None
    if use_radial_tukey_weight:
        r = rho.copy()
        a = float(tukey_alpha)
        w_r = np.ones_like(r)
        # Fade region: r in [1-a, 1]
        edge_region = (r >= 1 - a) & (r <= 1)
        w_r[edge_region] = 0.5 * (1 + np.cos(np.pi * (r[edge_region] - (1 - a)) / a))
        w_r[r > 1] = 0.0
        weights = w_r * aperture_mask.astype(float)

    projected_coefficients = None

    # Optional: project out specified Zernike modes
    if zero_zernike_indices is not None:
        remove_indices = [i for i, j in enumerate(indices) if j in zero_zernike_indices]
        if remove_indices:
            projection_mask = aperture_mask & np.isfinite(phase_processed)
            mask_flat = projection_mask.flatten()
            if np.count_nonzero(mask_flat) < len(remove_indices):
                raise ValueError(
                    "Not enough valid data points to project out the requested "
                    f"Zernike modes: {np.count_nonzero(mask_flat)} valid points "
                    f"for {len(remove_indices)} modes"
                )
            B_remove = np.array(
                [basis[i].flatten()[mask_flat] for i in remove_indices]
            ).T
            y_vec = phase_processed.flatten()[mask_flat]
            if weights is not None:
                w = weights.flatten()[mask_flat]
                w = np.clip(w, 0.0, None)
                sqrt_w = np.sqrt(w)
                B_remove = B_remove * sqrt_w[:, None]
                y_vec = y_vec * sqrt_w
            coeffs_remove, _, _, _ = np.linalg.lstsq(B_remove, y_vec, rcond=None)
            projected_coefficients = {
                indices[idx_rm]: float(coeff)
                for coeff, idx_rm in zip(coeffs_remove, remove_indices)
            }
            removal_map = np.zeros_like(phase_processed)
            for coeff, idx_rm in zip(coeffs_remove, remove_indices):
                removal_map += coeff * basis[idx_rm]
            phase_processed = phase_processed - removal_map

    # Fit Zernike polynomials. When low-order modes were explicitly projected
    # out, fit only the remaining modes so they cannot leak back in through the
    # discrete weighted solve.
    if zero_zernike_indices is not None:
        fit_positions = [i for i, j in enumerate(indices) if j not in zero_zernike_indices]
    else:
        fit_positions = list(range(len(indices)))

    fit_basis = [basis[i] for i in fit_positions]
    fit_indices = [indices[i] for i in fit_positions]

    coefficients_fit, fitted_phase, residual = fit_zernike_polynomials(
        phase_processed, fit_basis, fit_indices, weights=weights
    )

    if zero_zernike_indices is not None:
        coefficients = np.zeros(len(indices), dtype=np.float64)
        for coeff, pos in zip(coefficients_fit, fit_positions):
            coefficients[pos] = coeff
    else:
        coefficients = coefficients_fit

    # Calculate RMS error (use nanmean to handle NaN values)
    residual_valid = residual[aperture_mask]
    rms_error = np.sqrt(np.nanmean(residual_valid**2))
    rms_error_nm = float(
        np.sqrt(np.nanmean((residual_valid * wavelength / (2 * np.pi)) ** 2)) * 1e9
    )

    # Create results dictionaries
    zernike_terms = {f"Z{j}": float(coeff) for j, coeff in zip(indices, coefficients)}
    aberration_analysis = analyze_aberrations(coefficients, indices, wavelength)

    # Save results if directory provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        results = {
            "parameters": {
                "size": size,
                "wavelength_nm": wavelength * 1e9,
                "num_terms": num_terms,
                "aperture_radius_fraction": aperture_radius_fraction,
            },
            "fitted_coefficients": zernike_terms,
            "rms_error_rad": float(rms_error),
            "rms_error_nm": float(rms_error_nm),
            "aberration_analysis": aberration_analysis,
        }
        if projected_coefficients is not None:
            results["projected_out_coefficients"] = projected_coefficients
        with open(os.path.join(save_dir, "zernike_fit_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    return (
        coefficients,
        fitted_phase,
        residual,
        rms_error,
        zernike_terms,
        aberration_analysis,
    )


def visualize_zernike_analysis(
    phase,
    fitted_phase,
    residual,
    aberration_analysis,
    wavelength,
    pixel_size,
    title="Zernike Aberration Analysis",
    save_path=None,
    max_display_terms=None,
):
    """
    Visualize Zernike fitting results with comprehensive plots.

    Parameters:
    -----------
    phase : ndarray
        Original phase map.
    fitted_phase : ndarray
        Zernike-fitted phase.
    residual : ndarray
        Fitting residual (phase - fitted_phase).
    aberration_analysis : dict
        Dictionary of aberration analysis results.
    wavelength : float
        Wavelength in meters.
    pixel_size : tuple
        (py, px) pixel size in meters.
    title : str
        Plot title.
    save_path : str, optional
        Directory path to save the visualization figure. The image will be saved as
        'zernike_analysis.png' in this directory. If None, figure is not saved.
        Default is None.
    max_display_terms : int, optional
        Maximum number of Zernike terms to display in bar chart. If None,
        displays all fitted terms. Default is None.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Sort aberrations by OSA index (order)
    sorted_aberrations = sorted(
        aberration_analysis.items(),
        key=lambda x: x[1]["osa_index"],
    )

    if max_display_terms is None:
        num_display = len(sorted_aberrations)
    else:
        num_display = min(max_display_terms, len(sorted_aberrations))

    # Convert to nanometers for display
    phase_nm = phase * wavelength / (2 * np.pi) * 1e9
    fitted_nm = fitted_phase * wavelength / (2 * np.pi) * 1e9
    residual_nm = residual * wavelength / (2 * np.pi) * 1e9

    # Print diagnostics
    print("\n  Visualization diagnostics:")
    print(f"    Phase range: [{np.nanmin(phase_nm):.3f}, {np.nanmax(phase_nm):.3f}] nm")
    print(
        f"    Fitted range: [{np.nanmin(fitted_nm):.3f}, {np.nanmax(fitted_nm):.3f}] nm"
    )
    print(
        f"    Residual range: [{np.nanmin(residual_nm):.3f}, {np.nanmax(residual_nm):.3f}] nm"
    )

    # Create figure with custom layout
    # Use width_ratios to give more space to the bar chart (left side)
    fig_height = max(12.0, 6.0 + 0.28 * num_display)
    fig = plt.figure(figsize=(18, fig_height))
    gs = GridSpec(2, 3, figure=fig)

    # Color limits
    vmin_phase = np.nanmin(phase_nm)
    vmax_phase = np.nanmax(phase_nm)
    vmax_res = max(abs(np.nanmin(residual_nm)), abs(np.nanmax(residual_nm)))

    # Extent for plots
    if isinstance(pixel_size, (tuple, list)):
        py, px = pixel_size
    else:
        py = px = pixel_size

    h, w = phase.shape
    extent = (-w * px * 1e3 / 2, w * px * 1e3 / 2, -h * py * 1e3 / 2, h * py * 1e3 / 2)

    # Plot 1: Original Phase
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        phase_nm,
        cmap="RdBu_r",
        extent=extent,
        vmin=vmin_phase,
        vmax=vmax_phase,
        origin="lower",
    )
    ax1.set_title("Phase Error")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    plt.colorbar(im1, ax=ax1, label="Phase (nm)", fraction=0.046, pad=0.04)

    # Plot 2: Fitted Phase
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        fitted_nm,
        cmap="RdBu_r",
        extent=extent,
        vmin=vmin_phase,
        vmax=vmax_phase,
        origin="lower",
    )
    ax2.set_title("Zernike Fit")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    plt.colorbar(im2, ax=ax2, label="Phase (nm)", fraction=0.046, pad=0.04)

    # Plot 3: Residual
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(
        residual_nm,
        cmap="RdBu_r",
        extent=extent,
        vmin=-vmax_res,
        vmax=vmax_res,
        origin="lower",
    )
    ax3.set_title("Residual")
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("y (mm)")
    plt.colorbar(im3, ax=ax3, label="Residual (nm)", fraction=0.046, pad=0.04)

    # Plot 4: Aberration Bar Chart
    ax4 = fig.add_subplot(gs[1, :2])

    list_aberrations = sorted_aberrations[:num_display]
    labels = [f"{aberr['name']}:Z{aberr['osa_index']}" for _, aberr in list_aberrations]
    rms_values = [aberr["rms_nm"] for _, aberr in list_aberrations]

    ax4.barh(range(len(labels)), rms_values, color="steelblue", edgecolor="black")
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=9)
    ax4.set_xlabel("RMS Contribution (nm)")
    ax4.set_title(f"Zernike Coefficients (all {num_display} terms, by order)")
    ax4.grid(axis="x", alpha=0.3)
    ax4.invert_yaxis()

    # Plot 5: Statistics Table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    # Calculate statistics
    rms_total = np.sqrt(np.nanmean(residual**2))
    rms_total_nm = rms_total * wavelength / (2 * np.pi) * 1e9
    pv_nm = (np.nanmax(residual) - np.nanmin(residual)) * wavelength / (2 * np.pi) * 1e9

    stats_text = f"""
    FITTING STATISTICS
    {"=" * 30}
    
    RMS Error:
      {rms_total:.4f} rad
      {rms_total_nm:.3f} nm
      λ/{wavelength * 1e9 / rms_total_nm:.1f}
    
    Peak-to-Valley:
      {pv_nm:.3f} nm
    
    Number of Terms: {len(aberration_analysis)}
    
    Top 3 Aberrations:
    """
    # Sort by RMS descending to get the largest contributors
    top_aberrations = sorted(
        aberration_analysis.items(),
        key=lambda x: x[1]["rms_nm"],
        reverse=True,  # Largest first
    )

    for i, (key, aberr) in enumerate(top_aberrations[:3]):
        stats_text += f"\n  {i + 1}. Z{aberr['osa_index']} ({aberr['name']})"
        stats_text += f"\n     {aberr['rms_nm']:.3f} nm"

    ax5.text(
        0.1,
        0.9,
        stats_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Save figure if save_path is provided
    if save_path:
        img_path = os.path.join(save_path, "zernike_analysis.png")
        fig.savefig(img_path, dpi=300, bbox_inches="tight")
        print("MESSAGE: The Zernike analysis image is saved")
        print("-" * 50)

    plt.show()


def analyze_and_visualize_zernike(
    phase,
    pixel_size,
    wavelength,
    num_terms=36,
    save_dir=None,
    show_plots=True,
    save_path=None,
    **kwargs,
):
    """
    Complete Zernike analysis with visualization and formatted output.

    This is a high-level wrapper function that performs Zernike fitting,
    displays formatted results, and optionally shows visualization plots.

    Parameters:
    -----------
    phase : ndarray
        2D phase map in radians.
    pixel_size : float or tuple
        Pixel size in meters.
    wavelength : float
        Wavelength in meters.
    num_terms : int
        Number of Zernike terms to fit (default: 36). This also determines
        how many terms are displayed in the visualization.
    save_dir : str, optional
        Directory to save results.
    show_plots : bool
        Whether to display visualization plots (default: True).
    save_path : str, optional
        Directory path to save the visualization figure. The image will be saved as
        'zernike_analysis.png' in this directory. If None, figure is not saved.
        Default is None.
    **kwargs : dict
        Additional arguments passed to perform_zernike_analysis.

    Returns:
    --------
    dict: Complete analysis results including coefficients and aberrations
    """
    print("\n" + "=" * 70)
    print("ZERNIKE ABERRATION ANALYSIS")
    print("=" * 70)

    # Perform Zernike fitting
    (
        coefficients,
        fitted_phase,
        residual,
        rms_error,
        zernike_terms,
        aberration_analysis,
    ) = perform_zernike_analysis(
        phase=phase,
        pixel_size=pixel_size,
        wavelength=wavelength,
        num_terms=num_terms,
        save_dir=save_dir,
        **kwargs,
    )

    # Display summary
    print("\nZernike fitting completed:")
    print(f"  Number of terms: {num_terms}")
    print(
        f"  RMS residual: {rms_error:.4f} rad = {rms_error * wavelength / (2 * np.pi) * 1e9:.3f} nm"
    )

    # Display top aberrations
    print("\nFitted Zernike terms:")
    print("-" * 70)

    # Sort by OSA index so the full coefficient list follows the standard order.
    sorted_aberrations = sorted(
        aberration_analysis.items(), key=lambda x: x[1]["osa_index"]
    )

    print(f"{'Index':<8} {'Name':<30} {'Coeff (nm)':<15} {'RMS (nm)':<12}")
    print("-" * 70)
    for key, aberr in sorted_aberrations:
        j = aberr["osa_index"]
        name = aberr["name"]
        coeff_nm = aberr["coefficient_nm"]
        rms_nm = aberr["rms_nm"]
        print(f"Z{j:<7} {name:<30} {coeff_nm:>12.3f}   {rms_nm:>10.3f}")

    print("=" * 70 + "\n")

    # Visualization
    if show_plots:
        visualize_zernike_analysis(
            phase,
            fitted_phase,
            residual,
            aberration_analysis,
            wavelength,
            pixel_size,
            title="Zernike Aberration Analysis",
            save_path=save_path,
            max_display_terms=None,
        )

    # Return comprehensive results
    return {
        "coefficients": coefficients,
        "fitted_phase": fitted_phase,
        "residual": residual,
        "rms_error": rms_error,
        "zernike_terms": zernike_terms,
        "aberration_analysis": aberration_analysis,
    }
