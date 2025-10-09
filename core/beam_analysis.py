import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import least_squares


def calculate_and_visualize_beam(phase, virtual_pixel_size, title="Beam Analysis"):
    """
    Calculate beam position and size based on phase distribution and visualize on the phase map.
    Determines beam size by calculating horizontal and vertical signal distribution, then calculating FWHM.


    Args:
        phase: Reconstructed phase
        virtual_pixel_size: Virtual pixel size [px_x, px_y] in meters
        params: System parameters dictionary
        title: Plot title
        wavelength: Wavelength (meters), used for through-focus analysis

    Returns:
        tuple: (beam_position, beam_size, cropped_phase, crop_indices)
    """
    height, width = phase.shape

    # Extract horizontal and vertical average profiles
    # Calculate average profiles along x and y directions
    horizontal_profile = np.mean(phase, axis=0)  # Average along y-axis (rows)
    vertical_profile = np.mean(phase, axis=1)  # Average along x-axis (columns)

    # Create physical coordinate axes (microns)
    x_coords = (np.arange(width) - width / 2) * virtual_pixel_size[0] * 1e6  # Convert to microns
    y_coords = (np.arange(height) - height / 2) * virtual_pixel_size[1] * 1e6  # Convert to microns

    def calculate_fwhm_gaussian(profile, coords):
        """
        Fit a 1D Gaussian to the profile using scipy.optimize.least_squares
        and calculate FWHM and 1/e^2 radius.

        Args:
            profile (ndarray): 1D intensity or phase profile.
            coords (ndarray): Coordinate axis (same length as profile).

        Returns:
            fwhm (float): Full width at half maximum (same unit as coords).
            w0 (float): 1/e^2 radius.
            popt (tuple): (amplitude, mean, sigma, offset)
        """

        # --- Normalize profile to [0, 1] ---
        p_min = np.min(profile)
        p_range = np.max(profile) - p_min
        if p_range > 0:
            y_norm = (profile - p_min) / p_range
        else:
            y_norm = profile.copy()

        # --- Gaussian model ---
        def gaussian(x, a, mu, sigma, off):
            return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + off

        # --- Residual function ---
        def residuals(params, x, y):
            return gaussian(x, *params) - y

        # --- Initial guess ---
        peak_idx = np.argmax(y_norm)
        mean_init = coords[peak_idx]
        data_range = abs(coords[-1] - coords[0])

        # If peak is near edge → allow bigger sigma
        if peak_idx < len(y_norm) * 0.2 or peak_idx > len(y_norm) * 0.8:
            sigma_init = data_range
        else:
            sigma_init = data_range / 4

        p0 = [1.0, mean_init, sigma_init, 0.0]

        # --- Parameter bounds ---
        bounds_lower = [0, coords[0] - data_range, data_range * 1e-3, -0.5]
        bounds_upper = [2.0, coords[-1] + data_range, data_range * 2, 1.0]

        # --- Fit using least_squares ---
        res = least_squares(
            residuals, p0, bounds=(bounds_lower, bounds_upper),
            args=(coords, y_norm), max_nfev=2000
        )

        if not res.success:
            # Fallback: simple half-max method
            half_max = np.min(profile) + p_range * 0.5
            idx = np.where(profile >= half_max)[0]
            if len(idx) > 1:
                fwhm = abs(coords[idx[-1]] - coords[idx[0]])
            else:
                fwhm = data_range * 0.1
            return fwhm, fwhm / 1.177, None

        # --- Extract fitted parameters ---
        a, mu, sigma, off = res.x

        # --- Compute FWHM and w0 ---
        fwhm = 2.355 * abs(sigma)
        w0 = fwhm / 1.177

        return fwhm, w0, (a, mu, sigma, off)

    # Calculate FWHM and w0 (1/e² radius) in horizontal and vertical directions (using Gaussian fitting)
    fwhm_x, w0_x, fit_params_x = calculate_fwhm_gaussian(horizontal_profile, x_coords)
    fwhm_y, w0_y, fit_params_y = calculate_fwhm_gaussian(vertical_profile, y_coords)

    # Get beam center position from Gaussian fitting results
    if fit_params_x is not None:
        amplitude_x, mean_x, sigma_x, offset_x = fit_params_x
        beam_x_um = mean_x  # Horizontal beam center (micrometers)
    else:
        beam_x_um = 0.0  # If fitting fails, use image center
        print("Horizontal Gaussian fitting failed, using image center")

    if fit_params_y is not None:
        amplitude_y, mean_y, sigma_y, offset_y = fit_params_y
        beam_y_um = mean_y  # Vertical beam center (micrometers)
    else:
        beam_y_um = 0.0  # If fitting fails, use image center
        print("Vertical Gaussian fitting failed, using image center")

    # Calculate beam size parameters
    # Print beam size results
    # print(f"Beam center: ({beam_x_um:.2f}, {beam_y_um:.2f}) µm")
    # print(f"FWHM: x = {fwhm_x:.2f} µm, y = {fwhm_y:.2f} µm")
    # print(f"w0 (1/e² radius): x = {w0_x:.2f} µm, y = {w0_y:.2f} µm")

    # # Plotting moved to helper function
    # plot_beam_visualization(phase=phase, virtual_pixel_size=virtual_pixel_size, beam_x_um=beam_x_um,
    #                         beam_y_um=beam_y_um, fwhm_x=fwhm_x, fwhm_y=fwhm_y,
    #                         fit_params_x=fit_params_x, fit_params_y=fit_params_y, title=title)

    # Return beam position and size
    # Based on one-dimensional Gaussian fitting parameters
    beam_position = (beam_x_um * 1e-6, beam_y_um * 1e-6)  # Convert from microns to meters
    beam_size = {
        'fwhm_x': fwhm_x * 1e-6,  # FWHM in x direction, convert to meters
        'fwhm_y': fwhm_y * 1e-6,  # FWHM in y direction, convert to meters
    }

    return beam_position, beam_size


def plot_beam_visualization(phase, virtual_pixel_size, beam_x_um, beam_y_um, fwhm_x, fwhm_y, fit_params_x, fit_params_y,
                            title="Beam Analysis"):
    """
    Plot phase map with beam center/size overlays and profile fits, plus a 3D surface.

    Args:
        phase: Reconstructed phase map (2D array)
        virtual_pixel_size: [px_x, px_y] in meters
        beam_x_um, beam_y_um: Beam center in micrometers
        fwhm_x, fwhm_y: FWHM in micrometers
        w0_x, w0_y: 1/e^2 radii in micrometers
        fit_params_x, fit_params_y: Gaussian fit params (amplitude, mean, sigma, offset) or None
        is_valley_x, is_valley_y: Whether the profile is valley-type
        title: Plot title
        params: Parameter dict, used for saving path (expects params['image_path'])
        save_fig_flag: Whether to save the figure
        save_file_suf: Suffix for the saved file name
    """

    height, width = phase.shape

    # Physical sizes (m) and coordinates (um)
    x_size = width * virtual_pixel_size[0]
    y_size = height * virtual_pixel_size[1]
    x_coords = (np.arange(width) - width / 2) * virtual_pixel_size[0] * 1e6
    y_coords = (np.arange(height) - height / 2) * virtual_pixel_size[1] * 1e6

    # Profiles
    horizontal_profile = np.mean(phase, axis=0)
    vertical_profile = np.mean(phase, axis=1)

    # Figure and extents
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    extent_x = [-x_size / 2 * 1e6, x_size / 2 * 1e6]
    extent_y = [-y_size / 2 * 1e6, y_size / 2 * 1e6]
    extent = [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]

    # Phase image
    im = axes[0, 0].imshow(phase, cmap='viridis', extent=extent, origin='lower')
    plt.colorbar(im, ax=axes[0, 0], label='Phase (rad)')

    # Beam center and ellipse (FWHM)
    axes[0, 0].plot(beam_x_um, beam_y_um, 'rx', markersize=10, label='Beam Center')
    axes[0, 0].axhline(y=beam_y_um, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=beam_x_um, color='r', linestyle='--', alpha=0.5)
    ellipse = plt.matplotlib.patches.Ellipse(
        (beam_x_um, beam_y_um), fwhm_x, fwhm_y, fill=False, color='r', linestyle='-', linewidth=2,
        label='Beam Size (FWHM)'
    )
    axes[0, 0].add_patch(ellipse)

    # Info box
    info_text = (f"Beam Center: ({beam_x_um:.3f}, {beam_y_um:.3f}) µm\n"
                 f"FWHM X: {fwhm_x:.3f} µm\n"
                 f"FWHM Y: {fwhm_y:.3f} µm\n")
    axes[0, 0].text(0.02, 0.98, info_text, transform=axes[0, 0].transAxes,
                    va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 0].set_title(title)
    axes[0, 0].set_xlabel('x (µm)')
    axes[0, 0].set_ylabel('y (µm)')
    axes[0, 0].legend(loc='lower right')

    # Horizontal profile subplot
    if fit_params_x is not None:
        amplitude_x, mean_x, sigma_x, offset_x = fit_params_x
        h_min, h_max = np.min(horizontal_profile), np.max(horizontal_profile)
        h_range = h_max - h_min
        normalized_h = (horizontal_profile - h_min) / h_range if h_range > 0 else horizontal_profile.copy()

        fitted_curve_x = amplitude_x * np.exp(-((x_coords - mean_x) ** 2) / (2 * sigma_x ** 2)) + offset_x
        axes[1, 0].plot(x_coords, normalized_h, 'bx', markersize=4, label='data (norm.)')
        axes[1, 0].plot(x_coords, fitted_curve_x, 'r--', linewidth=2, label='Gauss fit')
    else:
        axes[1, 0].plot(x_coords, horizontal_profile, 'bx', markersize=4, label='data')
    axes[1, 0].set_title(f'Horizontal focus lineout FWHM {fwhm_x:.3f} µm')
    axes[1, 0].set_xlabel('x (µm)')
    axes[1, 0].set_ylabel('Intensity (norm.)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Vertical profile subplot
    if fit_params_y is not None:
        amplitude_y, mean_y, sigma_y, offset_y = fit_params_y
        v_min, v_max = np.min(vertical_profile), np.max(vertical_profile)
        v_range = v_max - v_min
        normalized_v = (vertical_profile - v_min) / v_range if v_range > 0 else vertical_profile.copy()

        fitted_curve_y = amplitude_y * np.exp(-((y_coords - mean_y) ** 2) / (2 * sigma_y ** 2)) + offset_y
        axes[0, 1].plot(y_coords, normalized_v, 'bx', markersize=4, label='data (norm.)')
        axes[0, 1].plot(y_coords, fitted_curve_y, 'r--', linewidth=2, label='Gauss fit')
    else:
        axes[0, 1].plot(y_coords, vertical_profile, 'bx', markersize=4, label='data')
    axes[0, 1].set_title(f'Vertical focus lineout FWHM {fwhm_y:.3f} µm')
    axes[0, 1].set_xlabel('y (µm)')
    axes[0, 1].set_ylabel('Intensity (norm.)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3D surface subplot
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
    ax3d.plot_surface(x_mesh, y_mesh, phase, cmap='viridis', alpha=0.8)
    ax3d.set_title('3D Phase Surface')
    ax3d.set_xlabel('x (µm)')
    ax3d.set_ylabel('y (µm)')
    ax3d.set_zlabel('Phase (rad)')

    plt.tight_layout()

    plt.show()


def analyze_focus_sampling_from_beam(amplitude, dx, dy, wavelength, propagation_distance, beam_size=None):
    """
    Calculate focus sampling rate based on Gaussian cone beam divergence angle, supporting separate calculation for X and Y directions

    Args:
        amplitude: Amplitude distribution
        dx: Current pixel size in X direction (m)
        dy: Current pixel size in Y direction (m)
        wavelength: Wavelength (m)
        propagation_distance: Propagation distance (m)
        beam_size: Optional, dictionary containing 'fwhm_x' and 'fwhm_y', if not provided will estimate from amplitude

    Returns:
        tuple: (dx_focus_final_x, dx_focus_final_y, divergence_angle_x, divergence_angle_y, w0_x, w0_y)
               - Focus sampling rates, divergence angles, and beam waist radii in X and Y directions
    """
    # If beam_size not provided, estimate FWHM from amplitude
    assert beam_size is not None, "beam_size must be provided"

    fwhm_x = beam_size['fwhm_x']
    fwhm_y = beam_size['fwhm_y']

    # print(f"Current FWHM: fwhm_x={fwhm_x * 1e6:.2f} μm, fwhm_y={fwhm_y * 1e6:.2f} μm")

    # Use far-field approximation to calculate X and Y directions separately

    # X direction calculation
    w_z_x = fwhm_x / (np.sqrt(2 * np.log(2)))

    def equation_x(w0):
        return w0 ** 2 * (1 + (propagation_distance * wavelength / (np.pi * w0 ** 2)) ** 2) - w_z_x ** 2

    w0_x, = fsolve(equation_x, 100e-9)

    # Y direction calculation
    w_z_y = fwhm_y / (np.sqrt(2 * np.log(2)))

    def equation_y(w0):
        return w0 ** 2 * (1 + (propagation_distance * wavelength / (np.pi * w0 ** 2)) ** 2) - w_z_y ** 2

    w0_y, = fsolve(equation_y, 100e-9)

    # Calculate FWHM at focus
    fwhm_focus_x = np.sqrt(2 * np.log(2)) * w0_x
    fwhm_focus_y = np.sqrt(2 * np.log(2)) * w0_y

    # Calculate divergence angle (half angle)
    divergence_angle_x = wavelength / (np.pi * w0_x)
    divergence_angle_y = wavelength / (np.pi * w0_y)

    # print(f"Final result:")
    # print(
    #     f"  X: fwhm={fwhm_focus_x * 1e6:.3f} μm, w0={w0_x * 1e6:.3f} μm, divergence angle={divergence_angle_x * 1e6:.3f} μrad")
    # print(
    #     f"  Y: fwhm={fwhm_focus_y * 1e6:.3f} μm, w0={w0_y * 1e6:.3f} μm, divergence angle={divergence_angle_y * 1e6:.3f} μrad")

    # Calculate beam size and sampling requirements at focus
    magnification_factor_x = fwhm_x / fwhm_focus_x
    magnification_factor_y = fwhm_y / fwhm_focus_y

    dx_focus_x = dx / magnification_factor_x
    dy_focus_y = dy / magnification_factor_y

    # Solution 2: Angular resolution based method
    angular_resolution_x = wavelength / (amplitude.shape[1] * dx)  # X direction angular resolution
    angular_resolution_y = wavelength / (amplitude.shape[0] * dy)  # Y direction angular resolution

    dx_focus_angular_x = angular_resolution_x * propagation_distance
    dy_focus_angular_y = angular_resolution_y * propagation_distance

    # print(f"Focus sampling:")
    # print(f"  X: based on fwhm={dx_focus_x * 1e9:.2f} nm, based on angular={dx_focus_angular_x * 1e9:.2f} nm")
    # print(f"  Y: based on fwhm={dy_focus_y * 1e9:.2f} nm, based on angular={dy_focus_angular_y * 1e9:.2f} nm")
    # print("-" * 50)
    # Choose more conservative (smaller) sampling interval
    dx_focus_final_x = min(dx_focus_x, dx_focus_angular_x)
    dx_focus_final_y = min(dy_focus_y, dy_focus_angular_y)

    return dx_focus_final_x, dx_focus_final_y, divergence_angle_x, divergence_angle_y, w0_x, w0_y
