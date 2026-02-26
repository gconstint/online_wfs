import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def preprocess_phase_for_fitting(
    phase, intensity=None, mask_threshold=0.1, outlier_sigma=3.0
):
    """
    Preprocess phase map before fitting to ensure robustness.

    Steps:
    1. Masking: Mask out background noise based on intensity or simple ROI.
    2. Outlier Removal: Remove spikes greater than mean +/- sigma * std.
    3. Nan Handling: Ensure NaNs are handled (though fitting functions should handle them).

    Args:
        phase (np.ndarray): Input phase map.
        intensity (np.ndarray, optional): Intensity map for masking.
        mask_threshold (float): Threshold relative to max intensity to define valid region.
        outlier_sigma (float): Sigma threshold for outlier removal.

    Returns:
        np.ndarray: Preprocessed phase map with NaNs in invalid/outlier regions.
    """
    phase_clean = phase.copy()

    # 1. Masking
    if intensity is not None:
        # Normalize intensity
        norm_intensity = intensity / np.max(intensity)
        mask = norm_intensity > mask_threshold
        phase_clean[~mask] = np.nan
    else:
        # Fallback: Circular mask if no intensity provided
        H, W = phase.shape
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        radius = min(H, W) // 2 * 0.95
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        phase_clean[~mask] = np.nan

    # 2. Outlier Removal (Despiking)
    # Calculate statistics on valid pixels
    valid_pixels = phase_clean[np.isfinite(phase_clean)]
    if len(valid_pixels) > 0:
        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)

        # Identify outliers
        lower_bound = mean_val - outlier_sigma * std_val
        upper_bound = mean_val + outlier_sigma * std_val

        outlier_mask = (phase_clean < lower_bound) | (phase_clean > upper_bound)
        phase_clean[outlier_mask] = np.nan

        # Optional: Fill NaNs with median of neighbors for continuity (if needed for some fitters)
        # But for our robust fitter, NaNs are better than bad values.

    return phase_clean


def plot_phase_error_profiles(
    phase_error,
    pixel_size=None,
    wavelength=None,
    save_path=None,
):
    """
    Plot 2D maps of horizontal and vertical phase error distributions.

    This function creates two 2D heatmaps:
    - Horizontal distribution: Each row shows the phase error profile along x at that y position
    - Vertical distribution: Each column shows the phase error profile along y at that x position

    Args:
        phase_error (numpy.ndarray): 2D array of phase error (radians)
        pixel_size (float or tuple, optional): Pixel size (meters)
        wavelength (float, optional): Wavelength (meters), used for unit conversion
        title_prefix (str, optional): Title prefix
    """
    if pixel_size is None or wavelength is None:
        print(
            "Warning: pixel_size or wavelength not provided, cannot plot phase error."
        )
        return

    # Handle pixel size
    if isinstance(pixel_size, (float, int)):
        py = px = float(pixel_size)
    else:
        py, px = float(pixel_size[0]), float(pixel_size[1])

    # Convert phase error to nanometers
    # phase_error (rad) * wavelength / (2*pi) = path length error (m)
    phase_error_nm = phase_error * wavelength / (2 * np.pi) * 1e9

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Extent for plotting (mm)
    H, W = phase_error.shape
    x_range = W * px * 1e3
    y_range = H * py * 1e3

    # Calculate symmetric color limits
    limit = max(abs(np.nanmin(phase_error_nm)), abs(np.nanmax(phase_error_nm)))

    # Plot 1: Horizontal distribution (phase error as function of x for all y)
    # This is just the original phase_error_nm displayed with x as horizontal axis
    extent_h = [-x_range / 2, x_range / 2, -y_range / 2, y_range / 2]
    rms_h = np.sqrt(np.nanmean(phase_error_nm**2))
    pv_h = np.nanmax(phase_error_nm) - np.nanmin(phase_error_nm)
    im0 = axes[0].imshow(
        phase_error_nm,
        cmap="RdBu_r",
        extent=extent_h,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axes[0].set_title(
        f"Horizontal Phase Error Distribution\nRMS: {rms_h:.2f} nm, PV: {pv_h:.2f} nm"
    )
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0], label="Phase Error (nm)", fraction=0.046, pad=0.04)

    # Plot 2: Vertical distribution (phase error as function of y for all x)
    # This is the transposed view with y as horizontal axis
    extent_v = [-y_range / 2, y_range / 2, -x_range / 2, x_range / 2]
    rms_v = rms_h  # Same data, same statistics
    pv_v = pv_h
    im1 = axes[1].imshow(
        phase_error_nm.T,
        cmap="RdBu_r",
        extent=extent_v,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axes[1].set_title(
        f"Vertical Phase Error Distribution\nRMS: {rms_v:.2f} nm, PV: {pv_v:.2f} nm"
    )
    axes[1].set_xlabel("y (mm)")
    axes[1].set_ylabel("x (mm)")
    fig.colorbar(im1, ax=axes[1], label="Phase Error (nm)", fraction=0.046, pad=0.04)

    plt.suptitle("Related wavefront distortions")
    plt.tight_layout()
    if save_path:
        img_path = os.path.join(save_path, "phase_error.png")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print("MESSAGE: The phase error image is saved")
        print("-" * 50)
    plt.show()


def fit_parabolic_phase(
    phase: np.ndarray, pixel_size: float | tuple[float, float]
) -> tuple[list[float], np.ndarray, dict]:
    """Robust parabolic phase fitting.

    This function implements a weighted least‑squares fit of a 2‑D parabolic
    phase model.  It returns the fitted parameters, the reconstructed phase,
    and a diagnostics dictionary containing the final cost, number of
    iterations and RMS residual.

    Model::
        phase(x, y) ≈ A * [2*((x‑x0)^2/Rx^2 + (y‑y0)^2/Ry^2) - 1]
                     + tx*(x‑x0) + ty*(y‑y0) + phi0

    Args:
        phase: 2‑D array of phase values (radians). NaNs are ignored.
        pixel_size: Either a scalar (square pixels) or a tuple (py, px).

    Returns:
        fit_params: ``[x0, y0, Rx, Ry, A]``
        fitted_phase: Reconstructed phase on the original grid.
        diagnostics: ``{"cost": float, "nfev": int, "rms": float}``
    """
    # --- Coordinates and units ---
    if isinstance(pixel_size, (float, int)):
        py, px = float(pixel_size), float(pixel_size)
    else:
        py, px = float(pixel_size[0]), float(pixel_size[1])

    H, W = phase.shape
    y = (np.arange(H) - H / 2) * py
    x = (np.arange(W) - W / 2) * px
    X, Y = np.meshgrid(x, y)

    # --- Initial values (backward compatible: x0,y0,R,A) ---
    x0_init, y0_init, Rx_init, Ry_init, A_init = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Rationalize initial values
    field_size = max(W * px, H * py)

    max_R = 10.0 * field_size

    # Avoid R initial value being 0 or too small
    Rx_init = float(
        np.clip(Rx_init if Rx_init > 0 else field_size, field_size * 0.2, max_R)
    )
    Ry_init = float(
        np.clip(Ry_init if Ry_init > 0 else field_size, field_size * 0.2, max_R)
    )
    A_init = float(
        A_init if np.isfinite(A_init) else (np.nanmax(phase) - np.nanmin(phase)) / 2.0
    )

    # Elliptical paraboloid initial values: Rx=Ry=R_init; first-order tilt default 0; constant term takes phase mean
    tx_init = 0.0
    ty_init = 0.0
    phi0_init = float(np.nanmean(phase))

    # --- Weights: Gaussian edge decay & NaN handling ---
    # Initialise weight map of ones; NaNs become zero weight
    weights = np.ones_like(phase, dtype=np.float64)
    weights[np.isnan(phase)] = 0.0

    # Apply a smooth Gaussian fall‑off towards the edges to suppress
    # boundary artefacts.  The sigma is chosen as 40% of the half‑size of the
    # field (empirically works well for typical wavefront data).
    sigma_x = 0.4 * (W * px) / 2.0
    sigma_y = 0.4 * (H * py) / 2.0
    gx = np.exp(-((X) ** 2) / (2 * sigma_x**2))
    gy = np.exp(-((Y) ** 2) / (2 * sigma_y**2))
    gaussian_weights = gx * gy
    weights *= gaussian_weights

    # Ensure a minimum overall weight to avoid degenerate optimisation
    if np.count_nonzero(weights) < phase.size * 0.05:
        weights[...] = 1.0
        weights[np.isnan(phase)] = 0.0

    # --- Model and residuals ---
    # Note: For fitting stability, use Rx,Ry directly as denominator (keep >0 bounds)
    def model(params):
        x0, y0, Rx, Ry, A, tx, ty, phi0 = params
        # Prevent division by zero
        Rx = np.maximum(Rx, 1e-12)
        Ry = np.maximum(Ry, 1e-12)
        dx = X - x0
        dy = Y - y0
        quad = A * (2.0 * ((dx * dx) / (Rx * Rx) + (dy * dy) / (Ry * Ry)) - 1.0)
        lin = tx * dx + ty * dy
        return quad + lin + phi0

    def residuals(params):
        pred = model(params)
        # Weighted residuals (center=1, edge=0), then flatten
        res = (pred - phase) * weights
        # Handle NaNs in phase by setting residual to 0 where weight is 0
        res[np.isnan(res)] = 0
        return res.ravel()

    # --- Parameter vector and bounds ---
    p0 = np.array(
        [x0_init, y0_init, Rx_init, Ry_init, A_init, tx_init, ty_init, phi0_init],
        dtype=np.float64,
    )

    # Bounds: x0,y0 within field size; Rx,Ry in [field_size*0.2, max_R]; A near phase amplitude; tx,ty relaxed; phi0 in ±(phase_range)
    phase_range = float(np.nanmax(phase) - np.nanmin(phase) + 1e-12)
    lower = np.array(
        [
            -field_size,
            -field_size,
            field_size * 0.2,
            field_size * 0.2,
            -2.0 * phase_range,
            -1e6,
            -1e6,
            np.nanmin(phase) - 2 * phase_range,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            field_size,
            field_size,
            max_R,
            max_R,
            +2.0 * phase_range,
            1e6,
            1e6,
            np.nanmax(phase) + 2 * phase_range,
        ],
        dtype=np.float64,
    )

    # Clip initial values to bounds to avoid "initial value out of bounds" exception
    p0 = np.clip(p0, lower + 1e-12, upper - 1e-12)

    # --- Robust least squares ---
    res = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",  # robust to outliers and edge artefacts
        f_scale=1.0,
        max_nfev=200000,
        verbose=0,
    )
    # Diagnostics
    diagnostics = {
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "rms": float(np.sqrt(res.cost / np.count_nonzero(weights))),
    }

    # Results and reconstruction
    x0, y0, Rx, Ry, A, tx, ty, phi0 = res.x
    fitted = model(res.x)
    fitted_phase = fitted.reshape(phase.shape)

    fit_params = [float(x0), float(y0), Rx, Ry, float(A)]
    return fit_params, fitted_phase, diagnostics


def find_wavefront_center(phase, pixel_size, verbose=True):
    """Find the wavefront center using quadratic fitting (same as fit_parabolic_phase_fast).

    Parameters:
    -----------
    phase : np.ndarray
        2D phase array (radians)
    pixel_size : float | tuple
        Pixel size in meters (py, px) or scalar
    verbose : bool
        Whether to print status messages (default: True)

    Returns:
    --------
    tuple: (x0, y0) in meters (physical coordinates)
    """
    if isinstance(pixel_size, (float, int)):
        py = px = float(pixel_size)
    else:
        py, px = float(pixel_size[0]), float(pixel_size[1])

    H, W = phase.shape

    # Use same method as fit_parabolic_phase_fast
    # Create coordinate grid (centered at image center)
    y = (np.arange(H) - H / 2) * py
    x = (np.arange(W) - W / 2) * px
    X, Y = np.meshgrid(x, y)

    # Flatten arrays
    Xf, Yf, Pf = X.ravel(), Y.ravel(), phase.ravel()

    # Valid data mask (finite values only)
    valid_mask = np.isfinite(Pf)
    if not np.any(valid_mask):
        if verbose:
            print("  Center finding: No valid data, using image center")
        return 0.0, 0.0

    # Apply Gaussian weighting (same as fit_parabolic_phase_fast)
    sigma_x = 0.4 * (W * px) / 2.0
    sigma_y = 0.4 * (H * py) / 2.0
    gx = np.exp(-((Xf) ** 2) / (2 * sigma_x**2))
    gy = np.exp(-((Yf) ** 2) / (2 * sigma_y**2))
    w = gx * gy
    w[~valid_mask] = 0.0

    Xf = Xf[valid_mask]
    Yf = Yf[valid_mask]
    Pf = Pf[valid_mask]
    w = w[valid_mask]

    # Build design matrix for quadratic fitting: [x^2, y^2, x, y, constant]
    M = np.column_stack(
        [
            Xf**2,  # a
            Yf**2,  # b
            Xf,  # c
            Yf,  # d
            np.ones_like(Xf),  # f
        ]
    )

    # Weighted least squares
    W_mat = np.diag(w)
    coeffs, *_ = np.linalg.lstsq(W_mat @ M, W_mat @ Pf, rcond=None)
    a, b, c, d, f = coeffs

    # Extract center from quadratic coefficients
    x0 = -c / (2 * a) if abs(a) > 1e-14 else 0.0
    y0 = -d / (2 * b) if abs(b) > 1e-14 else 0.0

    if verbose:
        print(f"  Center finding (quadratic fit): ({x0 * 1e6:.2f}, {y0 * 1e6:.2f}) μm")

    return x0, y0


def fit_parabolic_phase_fast(
    phase: np.ndarray,
    pixel_size: float | tuple[float, float],
    fixed_center: tuple[float, float] | None = None,
    verbose: bool = True,
) -> tuple[list[float], np.ndarray, dict]:
    """Fast parabolic phase fitting using linear least‑squares.

    This version builds a quadratic polynomial model and solves it in a
    single step.  A Gaussian weighting (identical to the robust version) is
    applied to down‑weight edge pixels.  The function returns the fitted
    parameters, the reconstructed phase, and a diagnostics dictionary.

    Args:
        phase: 2‑D array of phase values (radians). NaNs are ignored.
        pixel_size: Scalar or ``(py, px)`` tuple describing pixel pitch.
        fixed_center: Optional (x0, y0) in meters. If provided, center is fixed during fitting.
        verbose: Whether to print status messages (default: True)

    Returns:
        fit_params: ``[x0, y0, Rx, Ry, A]``
        fitted_phase: Reconstructed phase on the original grid.
        diagnostics: ``{"rms": float, "valid_points": int}``
    """
    # --- Pixel size handling ---
    if isinstance(pixel_size, (float, int)):
        py = px = float(pixel_size)
    else:
        py, px = float(pixel_size[0]), float(pixel_size[1])

    # --- Coordinate grid (centered at image center) ---
    H, W = phase.shape
    y = (np.arange(H) - H / 2) * py
    x = (np.arange(W) - W / 2) * px
    X, Y = np.meshgrid(x, y)

    # --- Flatten arrays for linear algebra ---
    Xf, Yf, Pf = X.ravel(), Y.ravel(), phase.ravel()

    # Handle NaNs and apply Gaussian weighting
    valid_mask = np.isfinite(Pf)
    n_valid = np.sum(valid_mask)
    if n_valid < 10:
        return (
            [0, 0, np.inf, np.inf, 0],
            np.zeros_like(phase),
            {"rms": np.nan, "valid_points": int(n_valid)},
        )

    # Gaussian edge weighting (same sigma as robust version)
    sigma_x = 0.4 * (W * px) / 2.0
    sigma_y = 0.4 * (H * py) / 2.0
    gx = np.exp(-((Xf) ** 2) / (2 * sigma_x**2))
    gy = np.exp(-((Yf) ** 2) / (2 * sigma_y**2))
    w = gx * gy
    w[~valid_mask] = 0.0

    Xf = Xf[valid_mask]
    Yf = Yf[valid_mask]
    Pf = Pf[valid_mask]
    w = w[valid_mask]

    # --- Linear design matrix for quadratic fitting ---
    if fixed_center is not None:
        # Fixed center fitting: fit only a, b, f
        # Phase = a*(x-x0)^2 + b*(y-y0)^2 + f
        x0_fixed, y0_fixed = fixed_center
        Xf_shifted = Xf - x0_fixed
        Yf_shifted = Yf - y0_fixed

        M = np.column_stack(
            [
                Xf_shifted**2,  # a
                Yf_shifted**2,  # b
                np.ones_like(Xf),  # f
            ]
        )

        # Weighted least squares using broadcasting (O(N) instead of O(N²))
        # Original: W_mat @ M where W_mat = np.diag(w)
        # Optimized: w[:, None] * M (element-wise row scaling)
        M_weighted = w[:, None] * M
        Pf_weighted = w * Pf
        coeffs, *_ = np.linalg.lstsq(M_weighted, Pf_weighted, rcond=None)
        a, b, f = coeffs

        # Center is fixed
        x0, y0 = x0_fixed, y0_fixed
        if verbose:
            print(
                f"    Using fixed center in fitting: ({x0 * 1e6:.2f}, {y0 * 1e6:.2f}) μm"
            )

        # Reconstruct with fixed center: Phase = a*(X-x0)^2 + b*(Y-y0)^2 + f
        fitted = a * (X - x0) ** 2 + b * (Y - y0) ** 2 + f

    else:
        # Full fitting: fit a, b, c, d, f
        # Columns correspond to [x^2, y^2, x, y, constant]
        M = np.column_stack(
            [
                Xf**2,  # a
                Yf**2,  # b
                Xf,  # c
                Yf,  # d
                np.ones_like(Xf),  # f
            ]
        )

        # Weighted least squares using broadcasting (O(N) instead of O(N²))
        # Original: W_mat @ M where W_mat = np.diag(w)
        # Optimized: w[:, None] * M (element-wise row scaling)
        M_weighted = w[:, None] * M
        Pf_weighted = w * Pf
        coeffs, *_ = np.linalg.lstsq(M_weighted, Pf_weighted, rcond=None)
        a, b, c, d, f = coeffs

        # --- Convert coefficients back to physical parameters ---
        # Vertex (parabola centre)
        x0 = -c / (2 * a) if abs(a) > 1e-14 else 0.0
        y0 = -d / (2 * b) if abs(b) > 1e-14 else 0.0

        # Reconstruct fitted phase on the full grid
        fitted = a * X**2 + b * Y**2 + c * X + d * Y + f

    # Amplitude estimate from the range of the valid data
    A_est = (np.nanmax(Pf) - np.nanmin(Pf)) / 2.0
    if A_est == 0:
        A_est = 1.0

    # Radii of curvature
    Rx = np.sqrt(abs(2 * A_est / a)) if a != 0 else np.inf
    Ry = np.sqrt(abs(2 * A_est / b)) if b != 0 else np.inf
    if verbose:
        print("    Rx =", Rx, "m, Ry =", Ry, "m")

    # Diagnostics
    residuals = fitted - phase
    rms = float(np.sqrt(np.nanmean(residuals[np.isfinite(residuals)] ** 2)))
    diagnostics = {"rms": rms, "valid_points": int(n_valid)}

    fit_params = [float(x0), float(y0), float(Rx), float(Ry), float(A_est)]
    return fit_params, fitted, diagnostics


def plot_phase_fit_results(
    phase, fitted_phase, fitted_params, pixel_size=None, save_path=None
):
    """
    Visualize parabolic phase fitting results and errors.
    """
    # Calculate phase error
    phase_error = phase - fitted_phase  # rad

    rms_error = np.sqrt(np.nanmean(phase_error**2))
    pv_error = np.nanmax(phase_error) - np.nanmin(phase_error)

    # Convert to wavelength units
    rms_error_lambda = rms_error / (2 * np.pi)

    # Extract values from fitting parameters
    x0, y0, Rx, Ry, A = fitted_params

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    # If pixel size provided, calculate actual dimensions
    if pixel_size is not None:
        if isinstance(pixel_size, (float, int)):
            pixel_size_y = pixel_size_x = pixel_size
        else:
            pixel_size_y, pixel_size_x = pixel_size

        height, width = phase.shape
        x_size = width * pixel_size_x
        y_size = height * pixel_size_y

        # Use same coordinate system as user
        extent_x = [-x_size / 2 * 1e6, x_size / 2 * 1e6]  # Convert to micrometers
        extent_y = [-y_size / 2 * 1e6, y_size / 2 * 1e6]  # Convert to micrometers
        extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]

        # Set coordinate axis units
        for ax in axes:
            ax.set_xlabel("x (μm)")
            ax.set_ylabel("y (μm)")
    else:
        extent = None

    # Calculate phase dynamic range for unified color scale
    phase_min = np.nanmin(phase)
    phase_max = np.nanmax(phase)
    phase_range = phase_max - phase_min

    # Plot original phase
    im = axes[0].imshow(
        phase,
        cmap="viridis",
        extent=extent,
        vmin=phase_min,
        vmax=phase_max,
        origin="lower",
    )
    axes[0].set_title(f"Reconstructed Phase\nPV: {phase_range:.3f} rad")
    fig.colorbar(im, ax=axes[0], label="rad", fraction=0.046)

    # Plot fitted phase
    im = axes[1].imshow(
        fitted_phase,
        cmap="viridis",
        extent=extent,
        vmin=phase_min,
        vmax=phase_max,
        origin="lower",
    )
    fit_info = (
        f"Fitted Parabolic Phase\n"
        # f"Rx = {Rx * 1e6:.2f} μm, Ry = {Ry * 1e6:.2f} μm\n "
        f"A = {A:.2f} rad\n"
        f"x₀ = {x0 * 1e6:.2f} μm, y₀ = {y0 * 1e6:.2f} μm"
    )
    axes[1].set_title(fit_info, fontsize=10)
    fig.colorbar(im, ax=axes[1], label="rad", fraction=0.046)

    # Calculate error range for symmetric color scale
    error_abs_max = max(abs(np.nanmin(phase_error)), abs(np.nanmax(phase_error)))

    # Plot phase error
    im = axes[2].imshow(
        phase_error,
        cmap="RdBu_r",
        extent=extent,
        vmin=-error_abs_max,
        vmax=error_abs_max,
        origin="lower",
    )
    axes[2].set_title(
        f"Phase Error\n"
        f"RMSE: {rms_error:.3f} rad or {rms_error_lambda:.3f} λ\n"
        f"PV: {pv_error:.3f} rad"
    )
    fig.colorbar(im, ax=axes[2], label="rad", fraction=0.046)

    plt.tight_layout()
    if save_path:
        img_path = os.path.join(save_path, "phase_fit.png")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print("MESSAGE: The phase fit image is saved")
        print("-" * 50)

    plt.show()


def perform_wavefront_fitting(
    phase,
    virtual_pixel_size,
    wavelength,
    intensity=None,
    save_path=None,
    find_center_first=False,
    verbose=True,
    show_plots=True,
):
    """
    Perform wavefront fitting analysis with optional center finding.

    Args:
        phase: Reconstructed phase
        virtual_pixel_size: Virtual pixel size [px_x, px_y]
        wavelength: Wavelength
        intensity: Optional intensity map for masking
        save_path: Optional path to save visualization
        find_center_first: If True, find center before fitting (default: True)
        verbose: If True, print status messages (default: True)
        show_plots: If True, show plots (default: True)

    Returns:
        tuple: (fit_params, fitted_phase, phase_error) Fitting parameters, fitted phase and phase error
    """

    # 1. Preprocess Phase (Masking, Outlier Removal)
    phase_processed = preprocess_phase_for_fitting(phase, intensity=intensity)

    # # 2. Optional: Find center first
    # if find_center_first:
    #     if verbose:
    #         print("\n>>>Finding wavefront center...")
    #     x0_initial, y0_initial = find_wavefront_center(
    #         phase_processed,
    #         virtual_pixel_size,
    #         verbose=verbose,
    #     )
    #     if verbose:
    #         print(
    #             f"  Initial center estimate: ({x0_initial * 1e6:.2f}, {y0_initial * 1e6:.2f}) μm"
    #         )
    #     fixed_center = (x0_initial, y0_initial)
    # else:
    #     fixed_center = None
    # Default: do not use center point
    fixed_center = None
    # 3. Fit with fixed or free center
    fit_params, fitted_phase, diagnostics = fit_parabolic_phase_fast(
        phase_processed, virtual_pixel_size, fixed_center=fixed_center, verbose=verbose
    )

    if verbose:
        print("\n-== Parabolic Phase Fit Results ---")
        x0, y0, Rx, Ry, A = fit_params
        # Convert meters to micrometers for display
        print(f"Center: (x0, y0) = ({x0 * 1e6:.2f} μm, {y0 * 1e6:.2f} μm)")
        print(
            f"Virtual Radius: Rx = {Rx * 1e6:.2f} μm ({Rx:.6f} m) , Ry = {Ry * 1e6:.2f} μm ({Ry:.6f} m)"
        )
        print(f"Amplitude: A = {A:.2f} rad")
        print(f"Wavelength: λ = {wavelength:.3e} m = {wavelength * 1e9:.1f} nm")

        # Print diagnostics (different for fast vs robust fitting)
        if "cost" in diagnostics and "nfev" in diagnostics:
            # Robust fitting diagnostics
            print(
                f"Fit diagnostics – cost: {diagnostics['cost']:.3e}, "
                f"RMS residual: {diagnostics['rms']:.3e}, "
                f"iterations: {diagnostics['nfev']}"
            )
        else:
            # Fast fitting diagnostics
            print(
                f"Fit diagnostics – RMS residual: {diagnostics['rms']:.3e}, "
                f"valid points: {diagnostics.get('valid_points', 'N/A')}"
            )

        x0, y0, Rx, Ry, A = fit_params
        if A != 0:
            Rx_z = np.pi * Rx**2 / (2 * abs(A) * wavelength)
            Ry_z = np.pi * Ry**2 / (2 * abs(A) * wavelength)
            print(
                f"Standard formula: R_z = π*R²/(2*A*λ) = {Rx_z:.3e} m and {Ry_z:.3e} m"
            )

    # Calculate phase error using original phase (no NaN values)
    phase_error = phase - fitted_phase

    if verbose:
        # For RMS calculation, use processed phase to exclude outliers
        phase_error_processed = phase_processed - fitted_phase
        rms_error = np.sqrt(np.nanmean(phase_error_processed**2))
        print(f"RMS Error: {rms_error:.4f} rad")

    # Visualize results
    if show_plots:
        plot_phase_fit_results(
            phase, fitted_phase, fit_params, virtual_pixel_size, save_path
        )
        # For phase error profiles, use phase_error (same as returned value)
        plot_phase_error_profiles(
            phase_error, virtual_pixel_size, wavelength, save_path
        )

    return fitted_phase, phase_error, fit_params
