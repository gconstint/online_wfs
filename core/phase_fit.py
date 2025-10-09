import matplotlib.pyplot as plt
import numpy as np


def plot_phase_error_profiles(phase_error, pixel_size=None, wavelength=632.8e-9,
                              title_prefix="Related wavefront distortions"):
    """
    Fast plotting of horizontal and vertical profiles of phase error in nanometers,
    with λ/4 and λ/8 Rayleigh criterion reference lines.

    Args:
        phase_error (ndarray): 2D array of phase error (radians).
        pixel_size (float or tuple, optional): Pixel size in meters. Single value for square pixels.
        wavelength (float, optional): Wavelength in meters. Default = 632.8 nm.
        title_prefix (str, optional): Plot title prefix.

    Returns:
        (fig, ax): Matplotlib Figure and Axes.
    """
    # --- Precompute conversion factor (rad → nm) ---
    scale = wavelength * 1e9 / (2 * np.pi)

    # --- Dimensions and center ---
    H, W = phase_error.shape
    cy, cx = H // 2, W // 2

    # --- Extract profiles and convert to nm ---
    horizontal = phase_error[cy, :] * scale
    vertical = phase_error[:, cx] * scale

    # --- Pixel size handling ---
    if isinstance(pixel_size, (float, int)):
        py = px = float(pixel_size)
    else:
        py, px = float(pixel_size[0]), float(pixel_size[1])

    # --- Coordinates in mm ---
    x_coords = (np.arange(W) - W / 2) * px * 1e3
    y_coords = (np.arange(H) - H / 2) * py * 1e3

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plot profiles ---
    ax.plot(x_coords, horizontal, 'r-', lw=1.5, label="Horizontal")
    ax.plot(y_coords, vertical, 'b-', lw=1.5, label="Vertical")
    ax.set_xlabel("CCD detector (mm)")
    ax.set_ylabel("Wavefront distortion (nm)")
    ax.set_title(title_prefix)

    # --- Reference lines ---
    λ4 = wavelength * 1e9 / 4
    λ8 = wavelength * 1e9 / 8
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(λ4, color="k", ls="--", lw=0.8, alpha=0.7, label="λ/4")
    ax.axhline(-λ4, color="k", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(λ8, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(-λ8, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.axhspan(-λ8, λ8, color="green", alpha=0.1, label="λ/8 range")

    # --- Compute statistics (nm) ---
    h_rms = np.sqrt(np.mean(np.square(horizontal)))
    v_rms = np.sqrt(np.mean(np.square(vertical)))
    h_pv = horizontal.max() - horizontal.min()
    v_pv = vertical.max() - vertical.min()

    # --- Text box with results ---
    stats_text = (f"Horizontal: RMS={h_rms:.2f} nm, PV={h_pv:.2f} nm {'✓' if h_rms < λ4 else '✗'}\n"
                  f"Vertical:   RMS={v_rms:.2f} nm, PV={v_pv:.2f} nm {'✓' if v_rms < λ4 else '✗'}\n"
                  f"λ/4 = {λ4:.2f} nm")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


def fit_parabolic_phase(phase, pixel_size):
    """
    Fast parabolic phase fitting using linear least squares + parameter recovery.

    Model (approximation):
        phase(x,y) ≈ A * [ 2 * ( (x-x0)^2 / Rx^2 + (y-y0)^2 / Ry^2 ) - 1 ]
                      + tx*(x-x0) + ty*(y-y0) + phi0

    The model can be rewritten as a quadratic polynomial in x and y:
        phase(x,y) ≈ a*x^2 + b*y^2 + c*x + d*y + f
    which allows solving by a single linear least squares step.
    Then (x0, y0, Rx, Ry, A) are recovered from (a, b, c, d, f).

    Args:
        phase (ndarray): 2D phase distribution (radians), preferably unwrapped.
        pixel_size (float or tuple): Pixel size in meters.
                                     If float, assumes square pixels.
                                     If tuple, expects (py, px).

    Returns:
        fit_params (list): [x0, y0, Rx, Ry, A]
        fitted_phase (ndarray): Phase reconstructed from the fitted model.
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

    # --- Linear design matrix for quadratic fitting ---
    # Columns correspond to [x^2, y^2, x, y, constant]
    M = np.column_stack([
        Xf ** 2,  # coefficient a
        Yf ** 2,  # coefficient b
        Xf,  # coefficient c
        Yf,  # coefficient d
        np.ones_like(Xf)  # constant term f
    ])

    # --- Solve linear least squares: M * coeffs ≈ phase ---
    coeffs, *_ = np.linalg.lstsq(M, Pf, rcond=None)
    a, b, c, d, f = coeffs

    # --- Convert coefficients back to physical parameters ---
    # Vertex (parabola center shift)
    x0 = -c / (2 * a) if abs(a) > 1e-14 else 0.0
    y0 = -d / (2 * b) if abs(b) > 1e-14 else 0.0

    # Estimate amplitude A from dynamic range of phase
    A_est = (np.nanmax(Pf) - np.nanmin(Pf)) / 2.0
    if A_est == 0:
        A_est = 1.0

    # Radii of curvature (Rx, Ry) from quadratic coefficients
    Rx = np.sqrt(abs(2 * A_est / a)) if a != 0 else np.inf
    Ry = np.sqrt(abs(2 * A_est / b)) if b != 0 else np.inf

    # --- Reconstruct fitted phase from polynomial model ---
    fitted = (a * Xf ** 2 + b * Yf ** 2 + c * Xf + d * Yf + f).reshape(phase.shape)

    # --- Output ---
    fit_params = [float(x0), float(y0), float(Rx), float(Ry), float(A_est)]
    return fit_params, fitted


def plot_phase_fit_results(phase, fitted_phase, fitted_params, pixel_size=None, wavelength=None):
    """
    Visualize parabolic phase fitting results and errors.
    """
    # Calculate phase error
    phase_error = phase - fitted_phase
    rms_error = np.sqrt(np.mean(phase_error ** 2))
    # Convert to wavelength units
    rms_error_lambda = rms_error / (2 * np.pi)
    pv_error = np.max(phase_error) - np.min(phase_error)
    # Extract values from fitting parameters
    x0, y0, Rx, Ry, A = fitted_params

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
    else:
        extent = None

    # Calculate phase dynamic range for unified color scale
    phase_min = np.min(phase)
    phase_max = np.max(phase)
    phase_range = phase_max - phase_min

    # Plot original phase
    im = axes[0].imshow(phase, cmap='viridis', extent=extent, vmin=phase_min, vmax=phase_max)
    axes[0].set_title('Reconstructed Phase')
    fig.colorbar(im, ax=axes[0], label='rad', fraction=0.046)

    # Plot fitted phase
    im = axes[1].imshow(fitted_phase, cmap='viridis', extent=extent, vmin=phase_min, vmax=phase_max)
    fit_info = (f"Fitted Parabolic Phase\n"
                f"Rx = {Rx * 1e6:.2f} μm, Ry = {Ry * 1e6:.2f} μm\n "
                f"A = {A:.2f} rad\n"
                f"x₀ = {x0 * 1e6:.2f} μm, y₀ = {y0 * 1e6:.2f} μm")
    axes[1].set_title(fit_info, fontsize=10)
    fig.colorbar(im, ax=axes[1], label='rad', fraction=0.046)

    # Calculate error range for symmetric color scale
    error_abs_max = max(abs(np.min(phase_error)), abs(np.max(phase_error)))

    # Plot phase error
    im = axes[2].imshow(phase_error, cmap='RdBu_r', extent=extent,
                        vmin=-error_abs_max, vmax=error_abs_max)
    axes[2].set_title(
        f'Phase Error\n'
        f'RMSE: {rms_error:.3f} rad or {rms_error_lambda:.3f} λ\n'
        f'PV: {pv_error:.3f} rad'
    )
    fig.colorbar(im, ax=axes[2], label='rad', fraction=0.046)

    plt.tight_layout()

    plt.show()


def perform_wavefront_fitting(phase, virtual_pixel_size):
    """
    Perform wavefront fitting analysis

    Args:
        phase: Reconstructed phase
        virtual_pixel_size: Virtual pixel size [px_x, px_y]
        wavelength: Wavelength
        params: System parameters dictionary
        fit_method: Fitting method, options: 'spherical', 'parabolic', 'hyperbolic'

    Returns:
        tuple: (fit_params, fitted_phase, phase_error) Fitting parameters, fitted phase and phase error
    """

    fit_params, fitted_phase = fit_parabolic_phase(
        phase, virtual_pixel_size)

    # print("\n-== Parabolic Phase Fit Results ---")
    # x0, y0, Rx, Ry, A = fit_params
    # Convert meters to micrometers for display
    # print(f"Center: (x0, y0) = ({x0 * 1e6:.2f} μm, {y0 * 1e6:.2f} μm)")
    # print(f"Virtual Radius: Rx = {Rx * 1e6:.2f} μm ({Rx:.6f} m) Ry = {Ry * 1e6:.2f} μm ({Ry:.6f} m)")
    # print(f"Amplitude: A = {A:.2f} rad")
    # print(f"Wavelength: λ = {wavelength:.3e} m = {wavelength * 1e9:.1f} nm")

    # if A != 0:
    #     # Standard formula calculation
    #     Rx_z = np.pi * Rx ** 2 / (2 * abs(A) * wavelength)
    #     Ry_z = np.pi * Ry ** 2 / (2 * abs(A) * wavelength)
    # print(f"Standard formula: R_z = π*R²/(2*A*λ) = Rx_z = {Rx_z:.3e} m Ry_z = {Ry_z:.3e} m")

    # Calculate phase error
    phase_error = phase - fitted_phase
    # rms_error = np.sqrt(np.mean(phase_error ** 2))
    # print(f"RMS Error: {rms_error:.4f} rad")

    return fitted_phase, phase_error, fit_params
