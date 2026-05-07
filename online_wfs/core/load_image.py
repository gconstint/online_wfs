from os import cpu_count

import numpy as np
from scipy.fft import fft2, fftshift

from typing import Any, Dict, Optional
from .utils import load_images, image_correction
from .grating_analysis import calculate_harmonic_periods, accurate_harmonic_periods


def load_and_preprocess_image(
    params: Dict[str, Any],
    do_rotation: bool = True,
    rotation_angle: Optional[float] = None,
    img: Optional[np.ndarray] = None,
    dark: Optional[np.ndarray] = None,
    flat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
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
    Returns
    -------
    np.ndarray
        FFT of preprocessed image (img_fft)
    """

    def compute_fft(image: np.ndarray) -> np.ndarray:
        image32 = np.asarray(image, dtype=np.float32, order="C")
        return fftshift(fft2(image32, norm="ortho", workers=cpu_count()))

    # Load images from file or use provided arrays
    if img is None:
        # File path mode: load from disk
        img, dark, flat = load_images(
            params["image_path"], params["dark_image_path"], params["flat_image_path"]
        )
    # Apply dark field subtraction and flat field correction
    img = image_correction(img, flat=flat, dark=dark, epsilon=1e-8, normalize=False)

    # img_cropped = center_crop(img, target_size=crop_size)
    # don't crop the image
    img_cropped = img
    # Calculate theoretical harmonic periods
    harmonic_periods = calculate_harmonic_periods(
        (img_cropped.shape[0], img_cropped.shape[1]),
        params["pixel_size"],
        params["pattern_period"],
    )
    params["harmonic_periods"] = harmonic_periods

    # Image rotation correction to align peaks horizontally
    if not do_rotation:
        return compute_fft(img_cropped)

    if rotation_angle is not None:
        angle = rotation_angle
    else:
        img_fft_init = compute_fft(img_cropped)

        _, peak_positions = accurate_harmonic_periods(img_fft_init, harmonic_periods)

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

    if abs(angle) > 1e-6:
        import cv2

        rows, cols = img_cropped.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_cropped = cv2.warpAffine(img_cropped, rotation_matrix, (cols, rows))

    img_fft = compute_fft(img_cropped)

    return img_fft
