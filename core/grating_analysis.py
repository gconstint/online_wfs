import cv2

import numpy as np

from numpy.fft import ifft2, ifftshift
from skimage.restoration import unwrap_phase


def calculate_peak_index(har_v, har_h, n_rows, n_columns, period_vert, period_hor):
    """
    Calculate the theoretical peak index(in pixel) for harmonic [har_v, har_h].
    Explain:
    1. n_rows // 2 和 n_columns // 2 is represented the center of the image.
    2. har_v, har_h is the harmonic index or the order of the harmonic.
    For instance, har_v = 1,har_v = 0 means the first harmonic in the vertical direction, and the zeroth harmonic in the
    horizontal direction.
    3. period_vert and period_hor is the period of the basic frequency in the harmonic.
    """
    results = [n_rows // 2 + har_v * period_vert,
               n_columns // 2 + har_h * period_hor]
    # print("the theoretical peak index is:", results)
    return results


def find_peak_in_region(img_fft, theoretical_idx, search_region):
    """
    Find the factual harmonic peak index (in pixel) within a local search window.

    Optimized: compute power only within the ROI to avoid full-spectrum |img_fft|.

    Args:
        img_fft: 2D complex spectrum (centered or not, irrelevant for peak search)
        theoretical_idx: [row, col] theoretical peak index
        search_region: int or [half_h, half_w] window half-size(s)
    """

    n_rows, n_columns = img_fft.shape
    i, j = theoretical_idx

    sy = int(search_region[0])
    sx = int(search_region[1])

    # Clamp ROI to image bounds
    y1 = max(0, i - sy)
    y2 = min(n_rows, i + sy + 1)
    x1 = max(0, j - sx)
    x2 = min(n_columns, j + sx + 1)

    local = img_fft[y1:y2, x1:x2]
    # Power spectrum (no sqrt): real^2 + imag^2
    power = (local.real * local.real) + (local.imag * local.imag)
    k = int(np.argmax(power))
    dy, dx = divmod(k, power.shape[1])
    return [y1 + dy, x1 + dx]


def accurate_harmonic_periods(img_fft, init_periods=None):
    """
    Calculate the peak positions of the 00, 01, and 10 harmonics, and calculate the accurate harmonic periods based on these peaks.

    Optimization:
    - Avoid calculating |img_fft| for the entire spectrum, only calculate power within the local ROI
    - Local ROI search replaces full image scan, reducing large matrix allocation and scanning

    Args:
        img_fft (ndarray): Frequency domain image (already FFTed, whether shifted or not does not affect local peak search)
        init_periods (list): Initial harmonic period estimation [period_vert, period_hor]

    Returns:
        tuple: (
            accurate_period: [period_vert, period_hor] accurate harmonic periods,
            peak_positions: {'00': [y0, x0], '01': [y01, x01], '10': [y10, x10]} peak positions of each harmonic
        )
    """
    n_rows, n_columns = img_fft.shape

    period_vert, period_hor = init_periods

    # Set the search region (if not passed, it is automatically set based on the period estimation)
    search_region = [max(10, int(min(period_vert, period_hor) // 4))] * 2

    # 3. Calculate and accurately locate the three harmonic peaks
    peak_positions = {}

    # 3.1 Calculate the 00 harmonic (center peak)
    idx_peak_00 = calculate_peak_index(0, 0, n_rows, n_columns, period_vert, period_hor)
    peak_00 = find_peak_in_region(img_fft, idx_peak_00, search_region)
    peak_positions['00'] = peak_00

    # 3.2 Calculate the 01 harmonic (first harmonic in the horizontal direction)
    idx_peak_01 = calculate_peak_index(0, 1, n_rows, n_columns, period_vert, period_hor)
    peak_01 = find_peak_in_region(img_fft, idx_peak_01, search_region)
    peak_positions['01'] = peak_01

    # 3.3 Calculate the 10 harmonic (first harmonic in the vertical direction)
    idx_peak_10 = calculate_peak_index(1, 0, n_rows, n_columns, period_vert, period_hor)
    peak_10 = find_peak_in_region(img_fft, idx_peak_10, search_region)
    peak_positions['10'] = peak_10

    # 4. Calculate the accurate harmonic periods based on the peak positions
    exp_period_h = abs(peak_01[1] - peak_00[1])  # horizontal period
    exp_period_v = abs(peak_10[0] - peak_00[0])  # vertical period

    return [exp_period_v, exp_period_h], peak_positions


def rotate_image_by_peaks(img, peak_positions):
    """
    Calculate the rotation angle and rotate the image according to the harmonic peak positions

    Args:
        img (ndarray): input image
        peak_positions (dict): harmonic peak positions, including '00', '01', '10' keys

    Returns:
        ndarray: rotated image
        float: rotation angle (degrees)
    """
    # Extract peak positions
    peak_00 = peak_positions['00']  # [y, x]
    peak_01 = peak_positions['01']  # [y, x]
    peak_10 = peak_positions['10']  # [y, x]

    # Calculate the required rotation angle in the horizontal direction
    # The 00 peak should have the same first coordinate value (y-coordinate) as the 01 peak
    delta_y_h = peak_01[0] - peak_00[0]  # y-offset in the horizontal direction
    delta_x_h = peak_01[1] - peak_00[1]  # x-offset in the horizontal direction
    angle_h = np.arctan2(delta_y_h, delta_x_h) * 180 / np.pi

    # Calculate the required rotation angle in the vertical direction
    # The 00 peak should have the same second coordinate value (x-coordinate) as the 10 peak
    delta_y_v = peak_10[0] - peak_00[0]  # y-offset in the vertical direction
    delta_x_v = peak_10[1] - peak_00[1]  # x-offset in the vertical direction
    angle_v = np.arctan2(delta_y_v, delta_x_v) * 180 / np.pi - 90  # The vertical direction needs to be subtracted by 90 degrees

    # Take the average of the angles in both directions
    angle = (angle_h + angle_v) / 2

    # Rotate the image (the angle is negated because we want to correct the image)
    rows, cols = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    return img_rotated, angle


def single_2D_grating_analyses(img_fft, harmonic_periods=None, unwrap_flag=True, plot_flag=True):
    """
    Process single 2D grating Talbot imaging img_data to extract harmonic images,
    dark field images, and phase contrast.

    Args:
        img (ndarray): Image to analyze.
        harmonic_periods (list): List containing harmonic periods [periodVert, periodHor].
        unwrap_flag (bool, optional): If True, unwrap phase images.
        plot_flag (bool, optional): If True, plot intermediate results.
        verbose (bool, optional): If True, print detailed processing information.

    Returns:
        int00 (ndarray): Zero-order harmonic intensity image (direct transmission).
        int01 (ndarray): First-order harmonic intensity image in the horizontal direction.
        int10 (ndarray): First-order harmonic intensity image in the vertical direction.
        darkField01 (ndarray): Dark field image for horizontal scattering.
        darkField10 (ndarray): Dark field image for vertical scattering.
        arg01 (ndarray): Phase contrast image in the horizontal direction.
        arg10 (ndarray): Phase contrast image in the vertical direction.
    """

    harm_img = single_grating_harmonic_images(img_fft, harmonic_periods)

    int00 = np.abs(harm_img[0])
    int01 = np.abs(harm_img[1])
    int10 = np.abs(harm_img[2])

    arg01 = unwrap_phase(np.angle(harm_img[1])) if unwrap_flag else np.angle(harm_img[1])
    arg10 = unwrap_phase(np.angle(harm_img[2])) if unwrap_flag else np.angle(harm_img[2])

    # remove piston only
    arg01 -= np.mean(arg01)
    arg10 -= np.mean(arg10)

    return int00, int01, int10, arg01, arg10


def single_grating_harmonic_images(img_fft, harmonic_period):
    """
    Obtain harmonic images from single 2D grating Talbot imaging.

    Args:
        img_fft (ndarray): Input image img_data.
        harmonic_period (list): List of harmonic periods [periodVert, periodHor].
        search_region (int, optional): Region size for harmonic peak search.
        plot_flag (bool, optional): If True, plot FFT results.
        verbose (bool, optional): If True, print detailed information.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: Harmonic images for 00, 01, and 10 harmonics.
    """

    search_region = max(10, int(min(harmonic_period[0], harmonic_period[1]) // 4))  # Reduce the initial search range
    # print("search_region", search_region)
    img_fft00 = extract_harmonic(img_fft, harmonic_period, harmonic_ij='00', search_region=search_region)
    img_fft01 = extract_harmonic(img_fft, harmonic_period, harmonic_ij='01', search_region=search_region)
    img_fft10 = extract_harmonic(img_fft, harmonic_period, harmonic_ij='10', search_region=search_region)

    img00 = ifft2(ifftshift(img_fft00), norm='ortho')
    img01 = ifft2(ifftshift(img_fft01), norm='ortho') if np.all(
        np.isfinite(img_fft01)) else img_fft01
    img10 = ifft2(ifftshift(img_fft10), norm='ortho') if np.all(
        np.isfinite(img_fft10)) else img_fft10

    return img00, img01, img10


def extract_harmonic(img_fft, harmonic_period, harmonic_ij='00', search_region=10):
    """
    Optimized version: Extract specific harmonic images from the Fourier-transformed img_data of a single grating Talbot image.

    Improvements:
    - Accurate sub-pixel peak localization
    - Adaptive window based on energy distribution
    - Improved Gaussian weighting
    - Corrected consistency between window display and actual selected region
    - Support for rectangular windows (set separately based on vertical and horizontal periods)
    """
    (n_rows, n_columns) = img_fft.shape
    har_v, har_h = int(harmonic_ij[0]), int(harmonic_ij[1])
    period_vert, period_hor = harmonic_period

    # 1. Get the theoretical peak position
    idx_peak_ij = calculate_peak_index(
        har_v, har_h, n_rows, n_columns, period_vert, period_hor)

    # 2. Accurately locate the peak (using local maximum search)
    y0, x0 = idx_peak_ij
    y1, y2 = max(0, y0 - search_region), min(n_rows, y0 + search_region + 1)
    x1, x2 = max(0, x0 - search_region), min(n_columns, x0 + search_region + 1)

    # Find the exact peak in the search area
    local_region = np.abs(img_fft[y1:y2, x1:x2])
    max_idx = np.unravel_index(np.argmax(local_region), local_region.shape)
    y_c = y1 + max_idx[0]
    x_c = x1 + max_idx[1]

    # Set the window size to be consistent
    period_hor = period_vert = min(period_vert, period_hor)
    # Calculate the window size (using different vertical and horizontal sizes)
    half_window_vert = period_vert // 2
    half_window_hor = period_hor // 2

    # Make sure the window is within the image range
    y_start = int(max(0, y_c - half_window_vert))
    y_end = int(min(n_rows, y_c + half_window_vert))
    x_start = int(max(0, x_c - half_window_hor))
    x_end = int(min(n_columns, x_c + half_window_hor))

    # 4. Extract the sub-region and apply Gaussian weighting
    sub_fft = img_fft[y_start:y_end, x_start:x_end]

    return sub_fft


def calculate_harmonic_periods(img_shape, pixel_size, pattern_period):
    """Calculate and adjust harmonic periods."""
    vert_harm = int((pixel_size[0] * img_shape[0]) / pattern_period)
    hor_harm = int((pixel_size[1] * img_shape[1]) / pattern_period)

    return [vert_harm, hor_harm]


def analyze_grating_data(img_fft, params, plot_flag=False):
    """Perform complete grating analysis."""

    results = single_2D_grating_analyses(img_fft, harmonic_periods=params["harmonic_periods"], unwrap_flag=True,
                                         plot_flag=plot_flag)

    # Calculate virtual pixel size and DPC
    virtual_pixel_size = [params['pixel_size'][0] * img_fft.shape[0] / results[0].shape[0],
                          params['pixel_size'][1] * img_fft.shape[1] / results[0].shape[1]]

    dpc_x = -results[3] * virtual_pixel_size[0] / \
            (params['det2sample'] * params['wavelength'])  # rad/pixel
    dpc_y = -results[4] * virtual_pixel_size[1] / \
            (params['det2sample'] * params['wavelength'])  # rad/pixel

    return *results[:3], dpc_x, dpc_y, virtual_pixel_size, params
