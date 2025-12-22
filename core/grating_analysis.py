import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.fft import ifft2, ifftshift
from skimage.restoration import unwrap_phase
from typing import Tuple, List, Dict, Optional, Union, Any


# =============================================================================
# Helper Functions
# =============================================================================


def extent_func(img, pixel_size=1):
    """
    Calculate the extent of an image based on its size and pixel dimensions.

    Parameters
    ----------
    img : ndarray
        Image array whose extent is being calculated.
    pixel_size : float or list of floats, optional
        Pixel dimensions as [pixelsize_i, pixelsize_j]. If a single float is
        provided, both pixel dimensions are set to this value. Default is 1.

    Returns
    -------
    ndarray
        Array of extent coordinates (left, right, bottom, top) for the image.
    """
    # Ensure pixel_size is a list with two elements
    if isinstance(pixel_size, (int, float)):
        pixel_size = [pixel_size, pixel_size]

    # Calculate the extent
    half_shape_y, half_shape_x = img.shape[0] // 2, img.shape[1] // 2
    extent = np.array(
        [
            -half_shape_x * pixel_size[1],
            (img.shape[1] - half_shape_x) * pixel_size[1],
            -half_shape_y * pixel_size[0],
            (img.shape[0] - half_shape_y) * pixel_size[0],
        ]
    )

    return extent


def calculate_peak_index(
    har_v: int,
    har_h: int,
    n_rows: int,
    n_columns: int,
    period_vert: float,
    period_hor: float,
) -> List[float]:
    """
    Calculate the theoretical peak index (in pixels) for harmonic [har_v, har_h].

    Args:
        har_v (int): Harmonic index in the vertical direction.
        har_h (int): Harmonic index in the horizontal direction.
        n_rows (int): Number of rows in the image.
        n_columns (int): Number of columns in the image.
        period_vert (float): Period of the basic frequency in the vertical direction.
        period_hor (float): Period of the basic frequency in the horizontal direction.

    Returns:
        List[float]: Theoretical peak index [y, x].
    """
    # n_rows // 2 and n_columns // 2 represent the center of the spectrum (DC component)
    return [n_rows // 2 + har_v * period_vert, n_columns // 2 + har_h * period_hor]


def _idxPeak_ij_exp(
    img_fft: np.ndarray,
    har_v: int,
    har_h: int,
    period_vert: float,
    period_hor: float,
    search_region: int,
) -> List[int]:
    """
    Find the experimental (actual) peak index by searching around theoretical position.

    This matches WavePy's _idxPeak_ij_exp function.

    Args:
        img_fft: 2D complex FFT spectrum
        har_v: Harmonic index in vertical direction
        har_h: Harmonic index in horizontal direction
        period_vert: Vertical harmonic period in pixels
        period_hor: Horizontal harmonic period in pixels
        search_region: Half-size of search window around theoretical position

    Returns:
        [row, col]: Experimental peak indices
    """
    intensity = np.abs(img_fft)
    n_rows, n_columns = img_fft.shape

    # Get theoretical peak position
    idx_peak_ij = calculate_peak_index(
        har_v, har_h, n_rows, n_columns, period_vert, period_hor
    )

    # Direct ROI slicing (O(k²) instead of O(N²) with full mask array)
    idx_y, idx_x = int(idx_peak_ij[0]), int(idx_peak_ij[1])
    y1 = max(0, idx_y - search_region)
    y2 = min(n_rows, idx_y + search_region)
    x1 = max(0, idx_x - search_region)
    x2 = min(n_columns, idx_x + search_region)

    # Find maximum in local region
    local_region = intensity[y1:y2, x1:x2]
    local_max = np.unravel_index(np.argmax(local_region), local_region.shape)

    return [y1 + local_max[0], x1 + local_max[1]]


def _error_harmonic_peak(
    img_fft: np.ndarray,
    har_v: int,
    har_h: int,
    period_vert: float,
    period_hor: float,
    search_region: int = 10,
) -> Tuple[float, float]:
    """
    Calculate error between theoretical and experimental peak positions.

    This matches WavePy's _error_harmonic_peak function.

    Args:
        img_fft: 2D complex FFT spectrum
        har_v: Harmonic index in vertical direction
        har_h: Harmonic index in horizontal direction
        period_vert: Vertical harmonic period in pixels
        period_hor: Horizontal harmonic period in pixels
        search_region: Half-size of search window

    Returns:
        (del_i, del_j): Error in pixels (vertical, horizontal)
    """
    n_rows, n_columns = img_fft.shape

    # Theoretical peak position
    idx_peak_ij = calculate_peak_index(
        har_v, har_h, n_rows, n_columns, period_vert, period_hor
    )

    # Experimental peak position
    idx_peak_ij_exp = _idxPeak_ij_exp(
        img_fft, har_v, har_h, period_vert, period_hor, search_region
    )

    # Calculate error
    del_i = idx_peak_ij_exp[0] - idx_peak_ij[0]
    del_j = idx_peak_ij_exp[1] - idx_peak_ij[1]

    return del_i, del_j


def find_peak_in_region(
    img_fft: np.ndarray,
    theoretical_idx: List[float],
    search_region: Union[int, List[int]],
    intensity: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Find the actual harmonic peak index (in pixels) within a local search window.

    Note: This is a legacy function kept for compatibility.
    Consider using _idxPeak_ij_exp for new code.

    Args:
        img_fft (np.ndarray): 2D complex spectrum.
        theoretical_idx (List[float]): [row, col] theoretical peak index.
        search_region (Union[int, List[int]]): Search window half-size. Can be a single int or [half_h, half_w].
        intensity (Optional[np.ndarray]): Precomputed |img_fft| to avoid redundant calculation.

    Returns:
        List[int]: Actual peak index [y, x].
    """
    if intensity is None:
        intensity = np.abs(img_fft)

    n_rows, n_columns = intensity.shape
    i, j = theoretical_idx

    if isinstance(search_region, int):
        sy = sx = search_region
    else:
        sy, sx = search_region

    # Clamp ROI to image bounds
    y1 = int(max(0, i - sy))
    y2 = int(min(n_rows, i + sy + 1))
    x1 = int(max(0, j - sx))
    x2 = int(min(n_columns, j + sx + 1))

    local = intensity[y1:y2, x1:x2]
    local_max = np.unravel_index(np.argmax(local), local.shape)

    return [y1 + local_max[0], x1 + local_max[1]]


def calculate_harmonic_periods(
    img_shape: Tuple[int, int], pixel_size: Tuple[float, float], pattern_period: float
) -> List[float]:
    """
    Calculate theoretical harmonic periods based on image parameters.

    Args:
        img_shape (Tuple[int, int]): Shape of the image (rows, cols).
        pixel_size (Tuple[float, float]): Pixel size in meters (py, px).
        pattern_period (float): Pattern period in meters.

    Returns:
        List[float]: Harmonic periods [period_vert, period_hor] in pixels.
    """
    vert_harm = (pixel_size[0] * img_shape[0]) / pattern_period + 1
    hor_harm = (pixel_size[1] * img_shape[1]) / pattern_period + 1
    return [vert_harm, hor_harm]


# =============================================================================
# Core Analysis Functions
# =============================================================================


def extract_harmonic(
    img_fft: np.ndarray,
    harmonic_period: List[float],
    harmonic_ij: str = "00",
    search_region: int = 10,
    plot_flag: bool = True,
    verbose: bool = True,
    use_theoretical_peak: bool = True,
) -> np.ndarray:
    """
    Extract specific harmonic images from the Fourier-transformed data.

    This function matches WavePy's extract_harmonic approach:
    - Uses THEORETICAL peak position for extraction window centering (default)
    - Optionally validates with experimental peak search
    - Rectangular window support for non-square gratings

    Args:
        img_fft: Input Fourier spectrum (fftshifted)
        harmonic_period: Harmonic periods [period_vert, period_hor] in pixels
        harmonic_ij: Harmonic identifier (e.g., '00', '01', '10')
        search_region: Half-size of search region for peak validation
        plot_flag: Whether to plot the extraction process
        verbose: Whether to show detailed information
        use_theoretical_peak: If True, use theoretical peak for extraction (WavePy default).
                             If False, use experimental peak search.

    Returns:
        Extracted harmonic spectrum (cropped FFT region)
    """
    n_rows, n_columns = img_fft.shape
    har_v, har_h = int(harmonic_ij[0]), int(harmonic_ij[1])
    period_vert, period_hor = harmonic_period

    if verbose:
        print(f"MESSAGE: Extracting harmonic {harmonic_ij}")
        print(f"MESSAGE: Harmonic period Horizontal: {int(period_hor)} pixels")
        print(f"MESSAGE: Harmonic period Vertical: {int(period_vert)} pixels")

    # 1. Get theoretical peak position
    idx_peak_ij = calculate_peak_index(
        har_v, har_h, n_rows, n_columns, period_vert, period_hor
    )

    # 2. Calculate peak position error only when needed (verbose mode or experimental peak)
    # This saves computation when using theoretical peaks in non-verbose mode
    if verbose or not use_theoretical_peak:
        del_i, del_j = _error_harmonic_peak(
            img_fft, har_v, har_h, period_vert, period_hor, search_region
        )

        if verbose:
            print(
                f"MESSAGE: Theoretical peak index: {idx_peak_ij[0]},{idx_peak_ij[1]} [VxH]"
            )
            print(
                f"MESSAGE: Harmonic peak {harmonic_ij} is misplaced by: "
                f"{del_i} pixels in vertical, {del_j} pixels in horizontal"
            )

        # Warn if peak is far from theoretical position (WavePy compatibility)
        if (np.abs(del_i) > search_region // 2) or (np.abs(del_j) > search_region // 2):
            print(
                f"ATTENTION: Harmonic Peak {harmonic_ij} is too far from theoretical value.\n"
                f"ATTENTION: {del_i} pixels in vertical, {del_j} pixels in horizontal"
            )

    # 3. Determine extraction window size
    half_window_vert = int(period_vert // 2)
    half_window_hor = int(period_hor // 2)

    # 4. Choose peak position for extraction window centering
    if use_theoretical_peak:
        # WavePy approach: use theoretical peak for stability
        y_peak, x_peak = idx_peak_ij
    else:
        # Alternative: use experimental peak (can be affected by noise)
        idx_peak_exp = _idxPeak_ij_exp(
            img_fft, har_v, har_h, period_vert, period_hor, search_region
        )
        y_peak, x_peak = idx_peak_exp

    # 5. Extract sub-region
    y_start = int(max(0, y_peak - half_window_vert))
    y_end = int(min(n_rows, y_peak + half_window_vert))
    x_start = int(max(0, x_peak - half_window_hor))
    x_end = int(min(n_columns, x_peak + half_window_hor))

    sub_fft = img_fft[y_start:y_end, x_start:x_end]

    # 6. Apply circular Gaussian window to reduce spectral leakage
    # Use broadcasting instead of meshgrid for memory efficiency
    ny, nx = sub_fft.shape
    y_coords = np.linspace(-1, 1, ny)[:, None]  # (ny, 1)
    x_coords = np.linspace(-1, 1, nx)[None, :]  # (1, nx)
    r_sq = y_coords**2 + x_coords**2  # Broadcasting: (ny, nx)
    sigma_sq_2 = 0.5  # 2 * sigma^2 where sigma = 0.5
    gaussian_2d = np.exp(-r_sq / sigma_sq_2)
    sub_fft = sub_fft * gaussian_2d

    # Get peak value for plotting
    peak_val = np.abs(img_fft[int(y_peak), int(x_peak)])

    # 6. Visualization
    if plot_flag:
        _plot_extraction(
            img_fft,
            sub_fft,
            harmonic_ij,
            peak_val,
            (x_peak, y_peak),
            (x_start, x_end, y_start, y_end),
            n_rows,
            n_columns,
            verbose,
        )

    return sub_fft


def _plot_extraction(
    img_fft,
    sub_fft,
    harmonic_ij,
    peak_val,
    peak_pos,
    window_coords,
    n_rows,
    n_columns,
    verbose,
):
    """Helper function for visualizing harmonic extraction."""
    x_peak, y_peak = peak_pos  # Now using theoretical peak position
    x_start, x_end, y_start, y_end = window_coords

    plt.figure(figsize=(10, 8))
    plt.imshow(
        np.log10(np.abs(img_fft)), cmap="inferno", extent=extent_func(np.abs(img_fft))
    )

    # Mark theoretical peak position
    plt.scatter(
        [x_peak - n_columns // 2],
        [n_rows // 2 - y_peak],
        color="cyan",
        marker="+",
        s=100,
        label="Theoretical Peak",
    )

    # Mark window
    rect = Rectangle(
        (x_start - n_columns // 2, n_rows // 2 - y_end),
        x_end - x_start,
        y_end - y_start,
        lw=2,
        ls="--",
        color="red",
        fill=False,
        alpha=1,
    )
    plt.gca().add_patch(rect)

    plt.title(f"Harmonic {harmonic_ij} Region (Peak: {peak_val:.2e})")
    plt.colorbar(label="Log10 Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if verbose:
        plt.figure(figsize=(6, 5))
        plt.imshow(np.log10(np.abs(sub_fft)), cmap="inferno")
        plt.title(f"Extracted Region (Harmonic {harmonic_ij})")
        plt.colorbar(label="Log10 Magnitude")
        plt.show()


def accurate_harmonic_periods(
    img_fft: np.ndarray, init_periods: List[float]
) -> Tuple[List[float], Dict[str, List[int]]]:
    """
    Calculate accurate harmonic periods based on experimental peak positions.

    This is a high-level function that:
    1. Finds experimental positions for main harmonics (00, 01, 10)
    2. Calculates corrected harmonic periods from peak spacing
    3. Returns both periods and peak positions for rotation correction

    Typical workflow:
        >>> periods, peaks = accurate_harmonic_periods(img_fft, init_periods)
        >>> img_rotated, angle = rotate_image_by_peaks(img, peaks)

    Args:
        img_fft: Input Fourier spectrum (fftshifted)
        init_periods: Initial estimate [period_vert, period_hor] in pixels

    Returns:
        Tuple:
            - accurate_periods [period_vert, period_hor]: Corrected periods
            - peak_positions: {'00': [y, x], '01': [y, x], '10': [y, x]}
    """
    n_rows, n_columns = img_fft.shape
    period_vert, period_hor = init_periods

    # Set search region (adaptive)
    search_region = int(max(10, min(period_vert, period_hor) // 4))

    peak_positions = {}

    # Find experimental peak positions using new helper function
    # 1. Find 00 peak (DC component)
    peak_positions["00"] = _idxPeak_ij_exp(
        img_fft, 0, 0, period_vert, period_hor, search_region
    )

    # 2. Find 01 peak (Horizontal harmonic)
    peak_positions["01"] = _idxPeak_ij_exp(
        img_fft, 0, 1, period_vert, period_hor, search_region
    )

    # 3. Find 10 peak (Vertical harmonic)
    peak_positions["10"] = _idxPeak_ij_exp(
        img_fft, 1, 0, period_vert, period_hor, search_region
    )

    # 4. Calculate corrected periods from experimental peak spacing
    exp_period_h = abs(peak_positions["01"][1] - peak_positions["00"][1])
    exp_period_v = abs(peak_positions["10"][0] - peak_positions["00"][0])

    return [exp_period_v, exp_period_h], peak_positions


def calculate_rotation_angle_from_peaks(peak_positions: Dict[str, List[int]]) -> float:
    """
    Calculate the rotation angle to align grating axes based on harmonic peak positions.

    Args:
        peak_positions (Dict): Peak positions {'00', '01', '10'}.

    Returns:
        float: Rotation angle in degrees.
    """
    peak_00 = peak_positions["00"]
    peak_01 = peak_positions["01"]
    peak_10 = peak_positions["10"]

    # Horizontal angle
    delta_y_h = peak_01[0] - peak_00[0]
    delta_x_h = peak_01[1] - peak_00[1]
    angle_h = np.arctan2(delta_y_h, delta_x_h) * 180 / np.pi

    # Vertical angle
    delta_y_v = peak_10[0] - peak_00[0]
    delta_x_v = peak_10[1] - peak_00[1]
    angle_v = np.arctan2(delta_y_v, delta_x_v) * 180 / np.pi - 90

    # Average angle
    angle = (angle_h + angle_v) / 2

    return angle


# =============================================================================
# High-Level Processing Functions
# =============================================================================


def single_grating_harmonic_images(
    img_fft: np.ndarray,
    harmonic_period: List[float],
    plot_flag: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and reconstruct harmonic images (00, 01, 10).

    Args:
        img_fft (np.ndarray): Input spectrum.
        harmonic_period (List[float]): Harmonic periods.
        plot_flag (bool): Whether to plot intermediate results.
        verbose (bool): Detailed logging.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Reconstructed complex images (img00, img01, img10).
    """
    # Adaptive search region
    search_region = int(max(10, min(harmonic_period[0], harmonic_period[1]) // 4))

    # Extract harmonics
    img_fft00 = extract_harmonic(
        img_fft, harmonic_period, "00", search_region, plot_flag, verbose
    )
    img_fft01 = extract_harmonic(
        img_fft, harmonic_period, "01", search_region, plot_flag, verbose
    )
    img_fft10 = extract_harmonic(
        img_fft, harmonic_period, "10", search_region, plot_flag, verbose
    )

    if plot_flag:
        _plot_harmonic_spectra(img_fft00, img_fft01, img_fft10)

    # Inverse FFT to spatial domain
    # Use scipy.fft which is generally faster than numpy.fft
    img00 = ifft2(ifftshift(img_fft00), norm="ortho")
    img01 = ifft2(ifftshift(img_fft01), norm="ortho")
    img10 = ifft2(ifftshift(img_fft10), norm="ortho")

    return img00, img01, img10


def _plot_harmonic_spectra(fft00, fft01, fft10):
    """Helper to plot harmonic spectra."""
    data = [np.log10(np.abs(x)) for x in [fft00, fft01, fft10]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["00", "01", "10"]

    for dat, ax, title in zip(data, axes, titles):
        im = ax.imshow(
            dat,
            cmap="inferno",
            vmin=np.min(dat),
            vmax=np.max(dat),
            extent=extent_func(dat),
        )
        ax.set_title(f"Harmonic {title}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("FFT subsets - Intensity(log scale)", fontsize=18, weight="bold")
    plt.show()


def single_2D_grating_analyses(
    img_fft: np.ndarray,
    img_ref_fft: Optional[np.ndarray] = None,
    harmonic_period: Optional[List[float]] = None,
    unwrap_flag: bool = True,
    plot_flag: bool = True,
    verbose: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Perform full 2D grating analysis: extraction, phase retrieval, and dark field calculation.

    Args:
        img_fft (np.ndarray): Sample image spectrum.
        img_ref_fft (Optional[np.ndarray]): Reference image spectrum.
        harmonic_period (List[float]): Harmonic periods [period_vert, period_hor].
        unwrap_flag (bool): Whether to unwrap phase.
        plot_flag (bool): Whether to plot results.
        verbose (bool): Detailed logging.

    Returns:
        Tuple containing:
        - int00, int01, int10: Intensity images
        - dark_field01, dark_field10: Dark field images
        - arg01, arg10: Phase contrast images (radians)
    """
    # 1. Extract harmonics for sample
    harm_img = single_grating_harmonic_images(
        img_fft, harmonic_period, plot_flag=plot_flag, verbose=verbose
    )

    # 2. Process with or without reference
    if img_ref_fft is not None:
        harm_img_ref = single_grating_harmonic_images(
            img_ref_fft, harmonic_period, plot_flag=False, verbose=verbose
        )

        # Intensity normalization (Transmission)
        int00 = np.abs(harm_img[0]) / np.abs(harm_img_ref[0])
        int01 = np.abs(harm_img[1]) / np.abs(harm_img_ref[1])
        int10 = np.abs(harm_img[2]) / np.abs(harm_img_ref[2])

        # Phase difference
        phase01 = np.angle(harm_img[1]) - np.angle(harm_img_ref[1])
        phase10 = np.angle(harm_img[2]) - np.angle(harm_img_ref[2])
    else:
        int00 = np.abs(harm_img[0])
        int01 = np.abs(harm_img[1])
        int10 = np.abs(harm_img[2])

        phase01 = np.angle(harm_img[1])
        phase10 = np.angle(harm_img[2])

    # 3. Phase Unwrapping
    if unwrap_flag:
        arg01 = unwrap_phase(phase01)
        arg10 = unwrap_phase(phase10)
    else:
        arg01, arg10 = phase01, phase10

    arg01 -= int(np.round(np.mean(arg01 / np.pi))) * np.pi
    arg10 -= int(np.round(np.mean(arg10 / np.pi))) * np.pi

    # 4. Dark Field Calculation
    # Match wavepy's approach:
    # - With reference: dark_field = (|H_sample|/|H_ref|) / (|H_00_sample|/|H_00_ref|)
    #                               = visibility_sample / visibility_ref
    # - Without reference: dark_field = |H_harmonic| / |H_00| = visibility_sample
    if img_ref_fft is not None:
        # When we have reference, int01 and int10 already contain the normalization by reference
        # int01 = |H_sample_01| / |H_ref_01|
        # int00 = |H_sample_00| / |H_ref_00|
        # Therefore: dark_field = int01 / int00 = visibility_sample / visibility_ref
        dark_field01 = int01 / int00
        dark_field10 = int10 / int00
    else:
        # Without reference, calculate sample visibility only
        dark_field01 = np.abs(harm_img[1]) / np.abs(harm_img[0])
        dark_field10 = np.abs(harm_img[2]) / np.abs(harm_img[0])

    return int00, int01, int10, dark_field01, dark_field10, arg01, arg10


def analyze_grating_data(
    img_fft: np.ndarray,
    img_ref_fft: Optional[np.ndarray],
    params: Dict[str, Any],
    plot_flag: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[float],
    Dict[str, Any],
]:
    """
    Orchestrate the complete grating analysis pipeline.

    Args:
        img_fft (np.ndarray): Sample spectrum.
        img_ref_fft (Optional[np.ndarray]): Reference spectrum.
        params (Dict): System parameters.
        plot_flag (bool): Visualization flag.

    Returns:
        Tuple containing analysis results, DPC signals, virtual pixel size, and updated params.
    """
    results = single_2D_grating_analyses(
        img_fft,
        img_ref_fft=img_ref_fft,
        harmonic_period=params["harmonic_periods"],
        plot_flag=plot_flag,
        unwrap_flag=True,
        verbose=False,
    )

    # Calculate virtual pixel size (accounting for downsampling in harmonic extraction)
    # The extracted harmonic image size is smaller than the original image
    virtual_pixel_size = [
        params["pixel_size"][0] * img_fft.shape[0] / results[0].shape[0],
        params["pixel_size"][1] * img_fft.shape[1] / results[0].shape[1],
    ]

    # Calculate DPC (Differential Phase Contrast)
    # Formula: DPC = -Phase * VirtualPixel / (Distance * Wavelength)
    # Units: radians (deflection angle)
    # Note: arg01 (results[5]) corresponds to horizontal (x) direction -> use virtual_pixel_size[1]
    #       arg10 (results[6]) corresponds to vertical (y) direction -> use virtual_pixel_size[0]
    factor_x = virtual_pixel_size[1] / (params["det2sample"] * params["wavelength"])
    factor_y = virtual_pixel_size[0] / (params["det2sample"] * params["wavelength"])

    dpc_x = -results[5] * factor_x  # arg01 * factor_x (horizontal)
    dpc_y = -results[6] * factor_y  # arg10 * factor_y (vertical)

    return *results[:5], dpc_x, dpc_y, virtual_pixel_size, params
